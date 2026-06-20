import streamlit as st
import requests

# pip install streamlit
# streamlit run front.py

st.title("Application d'identifiaction de médicaments")

image = st.file_uploader("Déposez l'image du médicament à identifier ici:", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"])

with st.spinner(text="Identification en cours...", show_time=False): 
	if st.button('Identifier médicament'):
	
		response = requests.post("http://localhost:8000/predict", files={"file": image})
		if response.status_code == 200:
			r = response.json()
			if r['reconnu']:
				st.header(f':green[Médicament reconnu: {r['medicament']['nom_fr']}]')
				if   r["scores"]["final"] >= 75: conf_txt = "haute"
				elif r["scores"]["final"] >= 60: conf_txt = "bonne"
				else: conf_txt = "moyenne"
				st.write(f"Confiance: {conf_txt}")
				st.table(r["medicament"])
			else:
				st.error("Médicament non reconnu, veuillez réessayer\n\n"
                         "Veuillez reprendre la photo en vous assurant que :\n"
                         "- la boîte est bien cadrée\n"
                         "- la lumière est suffisante\n"
                         "- le texte est net et lisible\n"
)
		else:
        		st.error(f"Error: {response.json().get('detail')}")


