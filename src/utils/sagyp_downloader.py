import os
import requests
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

BASE_URL = "https://www.magyp.gob.ar/sitio/areas/estimaciones/_archivos/estimaciones/"
DOWNLOAD_DIR = Path(__file__).resolve().parents[2] / "data" / "external"
YEARS = [[21, 2021], [22, 2022], [23, 2023], [24, 2024], [25, 2025], [26, 2026]]

MONTHS_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

def generate_url_variants(yy, mm, dd, yyyy_list):
    """
    SAGyP folks love changing naming conventions randomly. 
    This generator tries all dirty combinations of separators and words.
    """
    separators = ["_", "-"]
    weekly_words = ["semanal", "Semanal"]

    for sep1, sep2, weekly, y in product(separators, separators, weekly_words, yyyy_list):
        yield f"{yy}{mm}{dd}{sep1}Informe {weekly} al {dd}{sep2}{mm}{sep2}{y}.pdf"

def download_single_file(url, target_path):
    """ downloads a file and handles generic crashes safely """
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                f.write(response.content)
            return True
        return False
    except Exception as e:
        print(f"[ERROR] details: {e}")
        return False

def download_all_pdfs():
    """ 
    Scrapes the SAGyP site looking for weekly reports.
    They usually publish on Thursdays, so we jump week by week.
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    # To replicate our experiment, today should be the last date we consulted
    #today = datetime.now()
    today=datetime(2026,5,28)

    for y_info in YEARS:
        year = y_info[1]
        current_date = datetime(year, 1, 1)

        # Find the first Thursday of the year
        while current_date.weekday() != 3:
            current_date += timedelta(days=1)

        while current_date.year == year and current_date <= today:
            yyyy = str(current_date.year)
            yy = yyyy[2:]
            mm = f"{current_date.month:02d}"
            dd = f"{current_date.day:02d}"

            folder_year = f"{yy}0000_{yyyy}"
            folder_month = f"{yy}{mm}00_{MONTHS_ES[current_date.month]}"
            downloaded = False

            for file_name in generate_url_variants(yy, mm, dd, y_info):
                full_url = f"{BASE_URL}{folder_year}/{folder_month}/{file_name}".replace(" ", "%20")
                target_path = os.path.join(DOWNLOAD_DIR, file_name)

                print(f"Trying: {file_name}")
                if download_single_file(full_url, target_path):
                    print(f"[OK] Saved to {target_path}")
                    downloaded = True
                    break

            if not downloaded:
                print(f"[NOT FOUND] Report for {dd}/{mm}/{yyyy}")

            current_date += timedelta(days=7)