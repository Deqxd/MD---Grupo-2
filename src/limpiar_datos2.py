import os, glob, html, unicodedata
import pandas as pd
from pathlib import Path

# ------------------ UTILIDADES ------------------
def normalize_text(s: str) -> str:
    """Quita tildes, decodifica HTML y normaliza a MAYÚSCULAS sin espacios extra."""
    if s is None:
        return ""
    s = html.unescape(str(s)).strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = " ".join(s.split())
    return s.upper()

def map_columns(df):
    """
    Detecta columnas clave con tolerancia a variaciones:
    - seg_opcion: 'ESCUELA SEGUNDA OPCION' o 'ESCUELA PROFESIONAL (SEGUNDA OPCION)'
    - observacion: 'OBSERVACION' / 'OBSERVACIÓN' (incluye HTML)
    - merito: MERITOE.P / MERITO E.P / MERITO EP / MERITOE.P ALCANZA VACANTE ...
    """
    # limpiar encabezados y decodificar HTML
    real_cols = [html.unescape(str(c)).strip() for c in df.columns]
    df.columns = real_cols
    norm_to_real = {normalize_text(c): c for c in real_cols}

    def find_first(predicate):
        for nrm, real in norm_to_real.items():
            if predicate(nrm):
                return real
        return None

    # segunda opción
    seg_opcion = find_first(lambda n:
        ("ESCUELA" in n and "SEGUNDA" in n and "OPCION" in n)
    )

    # observación
    observacion = find_first(lambda n: "OBSERVACION" in n)

    # mérito (distintas variantes)
    def is_merito(n: str) -> bool:
        if "MERITO" not in n:
            return False
        # evitar confundir con puntaje
        if "PUNTAJE" in n:
            return False
        # señales típicas: ".P", "E.P", "EP", o textos largos con 'ALCANZA'/'VACANTE'
        return any(tok in n for tok in [".P", "E P", " EP", "ALCANZA", "VACANTE"])

    candidatos_merito = sorted(
        [(nrm, real) for nrm, real in norm_to_real.items() if is_merito(nrm)],
        key=lambda x: len(x[0]), reverse=True
    )
    merito = candidatos_merito[0][1] if candidatos_merito else None

    return {"seg_opcion": seg_opcion, "observacion": observacion, "merito": merito}

# palabras clave para detectar "alcanzó vacante en segunda/segundo opción"
KW_SEGUNDA = ["ALCANZO", "VACANTE", "SEGUND"]  # SEGUND* cubre SEGUNDO/SEGUNDA

# ------------------ PROCESAMIENTO ------------------
def process_file(in_path: str, out_path: str):
    # Lectura tolerante
    try:
        df = pd.read_csv(in_path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(in_path, dtype=str, encoding="latin-1")

    # Mapear columnas
    colmap = map_columns(df)

    # 1) Eliminar columna de segunda opción
    seg = colmap.get("seg_opcion")
    if seg and seg in df.columns:
        df = df.drop(columns=[seg])

    obs_col = colmap.get("observacion")
    mer_col = colmap.get("merito")

    if obs_col and obs_col in df.columns:
        # Normalizamos observaciones
        obs_raw = df[obs_col]
        obs_norm = obs_raw.map(normalize_text)

        # ---------- Regla 1: alcanzó vacante en segunda opción ----------
        mask_segunda = obs_norm.apply(lambda x: all(k in x for k in KW_SEGUNDA))
        # vaciar mérito y observación
        if mer_col and mer_col in df.columns:
            df.loc[mask_segunda, mer_col] = ""
        df.loc[mask_segunda, obs_col] = ""

        # Recalcular obs tras el vaciado anterior
        obs_raw2 = df[obs_col]
        obs_norm2 = obs_raw2.map(normalize_text)

        # ---------- Regla 2: si OBSERVACIÓN es AUSENTE o nula/vacía, mérito vacío ----------
        mask_ausente = (obs_raw2.isna()) | (obs_norm2.eq("AUSENTE")) | (obs_norm2.eq(""))
        if mer_col and mer_col in df.columns:
            df.loc[mask_ausente, mer_col] = ""

    # Guardar
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✔ {in_path} → {out_path}")

# ------------------ MAIN ------------------
def main():
    raiz = Path(__file__).resolve().parent.parent / "datos_admision"
    carpetas_objetivo = ["2024-I", "2024-II"]

    for carpeta in carpetas_objetivo:
        entrada = raiz / carpeta
        salida  = raiz / f"{carpeta}-limpio"

        if not entrada.exists():
            print(f"⚠ No existe {entrada}, se omite.")
            continue

        archivos = glob.glob(str(entrada / "*.csv"))
        if not archivos:
            print(f"⚠ No hay CSV en {entrada}")
            continue

        print(f"\nProcesando {carpeta} ({len(archivos)} archivo(s))...")
        for f in archivos:
            out_path = salida / Path(f).name
            try:
                process_file(f, out_path)
            except Exception as e:
                print(f"✖ Error con {f}: {e}")

if __name__ == "__main__":
    main()
