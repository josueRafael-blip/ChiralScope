import streamlit as st
import itertools
import os
import zipfile
import tempfile
import io
import sys

# Configuración para evitar warnings de RDKit
import warnings
warnings.filterwarnings('ignore')

# Manejo de importación de RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    st.error("❌ RDKit no está instalado. Por favor instala RDKit para usar la funcionalidad de conversión a XYZ.")
    st.info("Instala con: pip install rdkit")
    RDKIT_AVAILABLE = False

def detectar_quiralidad(smiles: str):
    if not RDKIT_AVAILABLE:
        return False, "RDKit no disponible", []
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "SMILES inválido", []
        
        centros = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        
        if len(centros) == 0:
            return False, "Su molécula no es quiral", []
        else:
            return True, f"Su molécula es quiral. Se detectaron {len(centros)} posibles centros", centros
            
    except Exception as e:
        return False, f"Error al analizar la molécula: {str(e)}", []

def analizar_centros_existentes(smiles: str):
    centros_especificados = 0
    posiciones_at = []
    i = 0
    
    while i < len(smiles):
        if smiles[i] == "@":
            if i + 1 < len(smiles) and smiles[i+1] == "@":
                centros_especificados += 1
                posiciones_at.append(i)
                i += 2
            else:
                centros_especificados += 1
                posiciones_at.append(i)
                i += 1
        else:
            i += 1
    
    return centros_especificados, posiciones_at

def generar_estereoisomeros(smiles: str):
    posiciones = []
    i = 0
    while i < len(smiles):
        if smiles[i] == "@":
            if i + 1 < len(smiles) and smiles[i+1] == "@":
                posiciones.append((i, True))  # ya es @@
                i += 2
            else:
                posiciones.append((i, False))  # es @ simple
                i += 1
        else:
            i += 1
    
    n = len(posiciones)
    
    if n == 0:
        st.warning("⚠️ El SMILES no tiene centros quirales especificados con @ o @@. No se generarán isómeros.")
        return [], n
    elif n > 3:
        st.error("❌ El SMILES tiene más de 3 centros quirales. No se generarán isómeros.")
        return [], n
    
    combinaciones = list(itertools.product(["@", "@@"], repeat=n))
    resultados = []
    
    for comb in combinaciones:
        chars = list(smiles)
        offset = 0
        for (pos, era_doble), val in zip(posiciones, comb):
            real_pos = pos + offset
            if era_doble:
                chars[real_pos:real_pos+2] = list(val)
                offset += len(val) - 2
            else:
                chars[real_pos:real_pos+1] = list(val)
                offset += len(val) - 1
        resultados.append("".join(chars))
    
    return resultados, n

def smiles_to_xyz(smiles, mol_id):
    if not RDKIT_AVAILABLE:
        return None, "❌ RDKit no está disponible"
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, f"❌ Error: SMILES inválido {smiles}"
        
        mol = Chem.AddHs(mol)
        
        params = AllChem.ETKDGv3()
        params.randomSeed = 42  
        
        embed_result = AllChem.EmbedMolecule(mol, params)
        if embed_result != 0:
            params.useRandomCoords = True
            embed_result = AllChem.EmbedMolecule(mol, params)
            if embed_result != 0:
                return None, f"⚠️ No se pudo generar conformación 3D para {smiles}"
        
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except:
            pass
        
        conf = mol.GetConformer()
        xyz_content = f"{mol.GetNumAtoms()}\n{smiles}\n"
        
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            xyz_content += f"{atom.GetSymbol()} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}\n"
        
        return xyz_content, f"✅ Molécula {mol_id} procesada correctamente"
        
    except Exception as e:
        return None, f"❌ Error procesando {smiles}: {str(e)}"

def crear_archivo_zip(archivos_xyz):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in archivos_xyz.items():
            zip_file.writestr(filename, content)
    return zip_buffer.getvalue()


def main():
    st.set_page_config(
        page_title="ChiralScope - Visualizador de Quiralidad",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Simple dark theme tweaks
    st.markdown(
        """
        <style>
        .css-1d391kg {padding-top: 1rem;}
        .stApp { background-color: #0f1115; color: #dfe7ef; }
        .stButton>button { background-color: #1f6feb; color: white; }
        .stDownloadButton>button { background-color: #1f6feb; color: white; }
        .stMarkdown { color: #dfe7ef; }
        .block-container { padding: 1rem 2rem; }
        .sidebar .css-1d391kg { background-color: #0b0c0f; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header with logo and title
    col1, col2 = st.columns([0.12, 0.88])
    with col1:
        try:
            st.image("assets/logo.svg", width=72)
        except:
            st.markdown("🧭")
    with col2:
        st.markdown("<h1 style='margin:0; color:#e6f0ff'>ChiralScope</h1>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:-10px;color:#9fb3d8'>Analizador de quiralidad y generador de estereoisómeros</div>", unsafe_allow_html=True)

    # Sidebar info
    with st.sidebar:
        try:
            st.image("assets/logo.svg", width=140)
        except:
            st.markdown("**ChiralScope**")
        st.markdown("---")
        st.markdown("**Instrucciones rápidas**")
        st.markdown("""
        1. Ingresa un SMILES en la pestaña *Ingreso*.
        2. Ve a *Análisis* para ver la quiralidad y centros detectados.
        3. Si hay centros quirales, genera los estereoisómeros y descárgalos.
        4. Usa *Visualización 3D* para explorar la conformación (si está disponible).
        """)
        st.markdown("---")
        st.markdown("**Desarrollado con RDKit y Streamlit**")

    # Tabs layout
    tabs = st.tabs(["🧾 Ingreso", "🔬 Análisis", "📦 Descargas", "🧭 Visualización 3D"])
    # Tab: Input
    with tabs[0]:
        st.header("Ingreso de SMILES")
        smiles_input = st.text_input("Introduce aquí el SMILES", placeholder="Ejemplo: C[C@H](O)[C@@H](N)C")
        st.write("También puedes arrastrar un archivo `.smi` o pegar múltiples SMILES separados por salto de línea abajo")
        upload = st.file_uploader("Subir archivo .smi (opcional)", type=["smi","txt"])
        uploaded_smiles = None
        if upload is not None:
            try:
                uploaded_smiles = upload.read().decode("utf-8").strip()
                st.success("Archivo cargado correctamente")
            except Exception as e:
                st.error("Error al leer el archivo: " + str(e))
        # Keep the last non-empty input as the active SMILES
        active_smiles = smiles_input.strip() if smiles_input else (uploaded_smiles.splitlines()[0].strip() if uploaded_smiles else "")

    # Analysis tab: uses helper functions defined above in the file
    with tabs[1]:
        st.header("Análisis de Quiralidad")
        if active_smiles:
            st.subheader("SMILES activo:")
            st.code(active_smiles)
            es_quiral, mensaje_quiralidad, centros_detectados = detectar_quiralidad(active_smiles)
            centros_especificados, posiciones_at = analizar_centros_existentes(active_smiles)
            st.markdown("**Resultado general:**")
            if es_quiral:
                st.success(f"✅ {mensaje_quiralidad}")
            else:
                if "inválido" in mensaje_quiralidad.lower():
                    st.error(f"❌ {mensaje_quiralidad}")
                else:
                    st.warning(f"⚠️ {mensaje_quiralidad}")

            st.markdown("**Centros detectados por RDKit:**")
            if centros_detectados:
                for idx, chirality in centros_detectados:
                    tipo = str(chirality) if chirality else "Sin asignar"
                    st.write(f"• Átomo {idx}: {tipo}")
            else:
                st.info("No se detectaron centros quirales automáticamente.")

            st.markdown("**Centros especificados en SMILES (@ / @@):**")
            st.write(f"{centros_especificados} centros encontrados en posiciones {posiciones_at}")

            if centros_especificados > 0:
                if st.button("🔄 Generar estereoisómeros"):
                    with st.spinner("Generando estereoisómeros..."):
                        isomeros, n_centros = generar_estereoisomeros(active_smiles)
                        if isomeros:
                            st.success(f"Se generaron {len(isomeros)} estereoisómeros (centros: {n_centros})")
                            # show first few
                            for i, iso in enumerate(isomeros[:10]):
                                st.code(f"{i+1}. {iso}")
                        else:
                            st.info("No se generaron estereoisómeros.")
            else:
                st.info("Si tu molécula es quiral pero no tiene centros especificados, añade @ o @@ en el SMILES.")

        else:
            st.info("Introduce un SMILES en la pestaña Ingreso para comenzar el análisis.")

    # Downloads tab
    with tabs[2]:
        st.header("Descargas")
        if active_smiles:
            # If isomeros were generated earlier in session, try to reuse; otherwise offer conversion of single SMILES
            try:
                isomeros, n_centros = generar_estereoisomeros(active_smiles)
            except Exception:
                isomeros, n_centros = [], 0
            if isomeros:
                smi_content = "\n".join(isomeros)
                st.download_button("📥 Descargar .smi (estereoisómeros)", smi_content, file_name="chiral_isomers.smi")
                zip_bytes = crear_archivo_zip(isomeros)
                st.download_button("📦 Descargar ZIP con .smi y .xyz", data=zip_bytes, file_name="chiral_outputs.zip")
            else:
                # offer conversion of the single SMILES to XYZ
                try:
                    xyz = smiles_to_xyz(active_smiles)
                    if xyz:
                        st.download_button("📥 Descargar .xyz (SMILES único)", xyz, file_name="molecule.xyz")
                except Exception as e:
                    st.info("No se pudo convertir a XYZ: " + str(e))
        else:
            st.info("Introduce un SMILES para obtener archivos descargables.")

    # 3D Visualization tab (simple fallback to show the first XYZ text or message)
    with tabs[3]:
        st.header("Visualización 3D")
        try:
            if active_smiles:
                xyz = smiles_to_xyz(active_smiles)
                if xyz:
                    st.code(xyz[:1000])
                    st.info("Puedes copiar el contenido XYZ y usar py3Dmol en otros entornos para visualización interactiva.")
                else:
                    st.info("No se pudo generar XYZ para este SMILES.")
            else:
                st.info("Introduce un SMILES para generar la conformación 3D.")
        except Exception as e:
            st.error("Error al generar visualización 3D: " + str(e))

    st.markdown('---')
    st.markdown(\"\"\"\n<div style='text-align: center'>\n    <small>🧭 <strong>ChiralScope</strong> - Rebranding del proyecto original<br>\n    Interfaz modernizada para análisis de quiralidad</small>\n</div>\n\"\"\", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
