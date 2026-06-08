from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import gspread
from datetime import datetime
import pdfplumber
import pandas as pd
import re
import io
from rdkit import Chem

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_uplc_data(pdf_file):
    extracted_data = {
        "workorder_id": "Not Found", "cln": "Not Found", "target_rt": None,
        "target_purity": None, "crude_mass_g": 0.0, "mass_category": "<=2 g",
        "major_peaks_count": 0, "closest_impurity_dRt": None, 
        "baseline_separated": False, "uplc_ph": "Unknown pH",
        "target_mass": None
    }
    with pdfplumber.open(pdf_file) as pdf:
        page1 = pdf.pages[0]
        text_page1 = page1.extract_text()
        wo_match = re.search(r'SampleNameList:\s*(\d+)', text_page1)
        if wo_match: extracted_data["workorder_id"] = wo_match.group(1)
        cln_match = re.search(r'UPLC Analysis Report:\s*([A-Za-z0-9\-]+)', text_page1)
        if cln_match: extracted_data["cln"] = cln_match.group(1)
        if "Low pH" in text_page1 or "low pH" in text_page1: extracted_data["uplc_ph"] = "Low pH"
        elif "High pH" in text_page1 or "high pH" in text_page1: extracted_data["uplc_ph"] = "High pH"
        
        mass_match = re.search(r'Result:.*?@\s*([0-9.]+)\s*mg', text_page1)
        if mass_match:
            mass_mg = float(mass_match.group(1))
            extracted_data["crude_mass_g"] = mass_mg / 1000.0
            if extracted_data["crude_mass_g"] > 5.0: extracted_data["mass_category"] = ">5 g"
            elif 2.0 < extracted_data["crude_mass_g"] <= 5.0: extracted_data["mass_category"] = "5 g - 2 g"
            else: extracted_data["mass_category"] = "<=2 g"
            
        tables = page1.extract_tables()
        for table in tables:
            if not table or not table[0]: continue
            headers = [str(h).replace('\n', ' ').strip() if h else "" for h in table[0]]
            df = pd.DataFrame(table[1:], columns=headers)
            
            if "Role" in df.columns:
                df["Role_Clean"] = df["Role"].astype(str).str.replace('\n', ' ')
                target_df = df[df["Role_Clean"].str.contains("TARGET PRODUCT", case=False, na=False)]
                if not target_df.empty:
                    target_row = target_df.iloc[0]
                    if "RT (min)" in df.columns:
                        rt_val = str(target_row["RT (min)"]).split('\n')[0].strip()
                        extracted_data["target_rt"] = float(rt_val) if rt_val.replace('.','',1).isdigit() else None
                    if "Purity (%)" in df.columns:
                        pur_val = str(target_row["Purity (%)"]).replace('\n', '').strip()
                        extracted_data["target_purity"] = float(pur_val) if pur_val.replace('.','',1).isdigit() else None
                    
                    for col in df.columns:
                        if "Mass" in col and "g/mol" in col:
                            mass_str = str(target_row[col]).replace('\n', '').strip()
                            m = re.search(r'([0-9.]+)', mass_str)
                            if m:
                                extracted_data["target_mass"] = float(m.group(1))
                            break
                    break
                    
        if len(pdf.pages) > 1:
            page2 = pdf.pages[1]
            text_lines = page2.extract_text().split('\n')
            peak_lines = [line for line in text_lines if len(line.split()) >= 4 and line.split()[0].isdigit()]
            peak_rts = []
            for line in peak_lines:
                try:
                    parts = line.split()
                    peak_rts.append(float(parts[1]))
                    if float(parts[2]) > 10.0: extracted_data["major_peaks_count"] += 1
                except ValueError: continue 
            if extracted_data["target_rt"]:
                target = extracted_data["target_rt"]
                distances = [abs(target - rt) for rt in peak_rts if rt != target]
                if distances:
                    closest_dRt = min(distances)
                    extracted_data["closest_impurity_dRt"] = closest_dRt
                    if closest_dRt > 0.05: extracted_data["baseline_separated"] = True
    return extracted_data

def get_prep_gradient(rt):
    if rt is None: return "Unknown Method"
    if rt <= 0.6: return "Polar Method"
    elif 0.6 < rt <= 1.2: return "Mid-Polar Method"
    else: return "Non-Polar Method"

def get_flash_cartridge(mass_g):
    mass_mg = mass_g * 1000
    if mass_mg < 499: return "Biotage Sfär 60μm 5g"
    elif 499 <= mass_mg < 999: return "Biotage Sfär 60μm 10g"
    elif 999 <= mass_mg < 2499: return "Biotage Sfär 60μm 25g"
    elif 2499 <= mass_mg < 4999: return "Biotage Sfär 60μm 50g"
    elif 4999 <= mass_mg < 9999: return "Biotage Sfär 60μm 100g"
    elif 9999 <= mass_mg < 19999: return "Biotage Sfär 60μm 200g"
    else: return "Biotage Sfär 60μm 300g"

# --- NEW: Tables added to the Flash Routing Logic ---
def get_flash_method_from_smiles(smiles):
    # Defining the tables exactly as you provided
    method1_table = [
        {"cv": 0, "a": 100, "b": 0},
        {"cv": 2, "a": 100, "b": 0},
        {"cv": 6, "a": 70, "b": 30},
        {"cv": 5, "a": 50, "b": 50},
        {"cv": 5, "a": 30, "b": 70},
        {"cv": 3, "a": 0, "b": 100},
        {"cv": 5, "a": 0, "b": 100}
    ]
    
    method2_table = [
        {"cv": 0, "a": 100, "b": 0},
        {"cv": 3, "a": 100, "b": 0},
        {"cv": 15, "a": 90, "b": 10},
        {"cv": 5, "a": 80, "b": 20},
        {"cv": 5, "a": 80, "b": 20}
    ]

    default_method = ("Method 1: General Gradient", "Heptane / Hexane", "5% MeOH in DCM", method1_table)
    
    if not smiles or smiles.strip() == "":
        return default_method
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return default_method
            
        carboxylic_acid = Chem.MolFromSmarts('C(=O)[OH]')
        primary_amine = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]') 
        
        if mol.HasSubstructMatch(carboxylic_acid) or mol.HasSubstructMatch(primary_amine):
            return ("Method 2: Acid-Modified Gradient", "DCM", "MeOH + 0.1% Formic Acid", method2_table)
        else:
            return default_method
    except:
        return default_method

def predict_purification(uplc_data, stage, moves_in_tlc):
    mass = uplc_data["mass_category"]
    purity = uplc_data["target_purity"]
    rt = uplc_data["target_rt"]
    crude_mass_g = uplc_data["crude_mass_g"]
    
    if purity is None: 
        return "Manual Review Required", None, "Could not read the Target Purity."
    
    def format_prep(r): return "Prep-HPLC", get_prep_gradient(rt), r
    def format_flash(r): return "Normal Phase Flash", get_flash_cartridge(crude_mass_g), r
    def format_rp_flash(r): return "Reverse Phase Flash", get_flash_cartridge(crude_mass_g) + " (C18)", r

    would_be_prep = False
    reason = ""

    if stage == "Final":
        if purity > 40 and uplc_data["baseline_separated"] and moves_in_tlc:
            reason = "Final product with good purity, baseline separated, and moves well on TLC."
        else:
            would_be_prep = True
            reason = "Final product prioritized for maximum purity or requires tight separation."
            
    elif stage == "Intermediate":
        if not uplc_data["baseline_separated"]: 
            would_be_prep = True
            reason = "No baseline separation on UPLC."
        elif uplc_data["major_peaks_count"] > 3: 
            would_be_prep = True
            reason = "Complex mixture (>3 major impurities)."
        elif uplc_data["baseline_separated"] and moves_in_tlc: 
            reason = "Baseline separated and moves well on TLC."
        else:
            would_be_prep = True
            reason = "Default conservative recommendation (does not move on TLC or requires tight separation)."

    is_high_mass = mass in [">5 g", "5 g - 2 g"]

    if is_high_mass:
        if would_be_prep and stage == "Intermediate":
            return format_rp_flash(reason + " [Rerouted to RP-Flash due to mass > 2g]")
        else:
            return format_flash("High capacity normal-phase required for mass > 2g.")
    
    if would_be_prep: return format_prep(reason)
    else: return format_flash(reason)

def log_to_google_sheets(data, method):
    try:
        gc = gspread.service_account(filename="credentials.json")
        sheet = gc.open("Purification Logs").sheet1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_to_add = [
            current_time, data["workorder_id"], data["cln"], 
            data["crude_mass_g"], data["target_rt"], method
        ]
        sheet.append_row(row_to_add)
    except Exception:
        pass 

def search_google_sheets(query):
    try:
        gc = gspread.service_account(filename="credentials.json")
        sheet = gc.open("Purification Logs").sheet1
        
        rows = sheet.get_all_values()
        q = query.strip().lower()
        results = []
        
        for row in reversed(rows):
            if len(row) >= 6:
                wo = str(row[1]).strip().lower()
                cln = str(row[2]).strip().lower()
                
                if q in wo or q in cln:
                    results.append({
                        "date": row[0],
                        "workorder": row[1],
                        "cln": row[2],
                        "mass": row[3],
                        "rt": row[4],
                        "method": row[5]
                    })
                    
        if len(results) > 0:
            return {"found": True, "results": results}
        else:
            return {"found": False, "message": f"No records found containing '{query}'"}
            
    except Exception as e:
        return {"found": False, "message": f"Google Sheets Error: {str(e)}"}

@app.post("/predict")
async def run_prediction(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    stage: str = Form(...),
    moves_in_tlc: str = Form(...),
    smiles: str = Form("")
):
    pdf_bytes = io.BytesIO(await file.read())
    data = extract_uplc_data(pdf_bytes)
    moves_bool = True if moves_in_tlc == "Yes" else False
    
    method, conditions, reason = predict_purification(data, stage, moves_bool)
    background_tasks.add_task(log_to_google_sheets, data, method)

    prep_conds = None
    flash_conds = None
    
    if method == "Prep-HPLC":
        vol_ml = round((data["crude_mass_g"] * 1000) / 100, 1)
        if data.get("target_mass"):
            m_ion = int(round(data["target_mass"] + 1))
        else:
            m_ion = "N/A"
            
        prep_conds = {
            "ph": data["uplc_ph"].replace(" pH", ""),
            "gradient": get_prep_gradient(data["target_rt"]),
            "volume_ml": vol_ml,
            "mass_ion": m_ion
        }
    
    elif method == "Normal Phase Flash":
        # Extract the table along with the method info
        method_name, sol_a, sol_b, method_table = get_flash_method_from_smiles(smiles)
        flash_conds = {
            "cartridge": conditions,
            "method_name": method_name,
            "solvent_a": sol_a,
            "solvent_b": sol_b,
            "table": method_table
        }

    return {
        "method": method,
        "conditions": conditions,
        "reason": reason,
        "data": data,
        "prep_conditions": prep_conds,
        "flash_conditions": flash_conds 
    }

@app.get("/search")
def search_records(q: str):
    return search_google_sheets(q)