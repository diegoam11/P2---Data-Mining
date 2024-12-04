from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import Options
import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

# Configurar el navegador
options = Options()
driver = webdriver.Chrome(service=Service('./chromedriver.exe'), options=options)

# Navegar a la página web
driver.get("https://www.reddit.com/r/Lima_Peru/comments/1gfwap2/los_datos_de_3_millones_de_usuarios_de_interbank/")

# Esperar a que se cargue el contenido dinámico
time.sleep(10)  

# Buscar los divs que tienen el atributo slot="comment"
comentarios_divs = driver.find_elements(By.CSS_SELECTOR, '[slot="comment"]')

# Lista para almacenar los comentarios como objetos
comentarios = []

# Iterar sobre cada div encontrado y obtener el texto de los p dentro de él
for div in comentarios_divs:
    # Buscar todas las etiquetas <p> dentro de este div
    parrafos = div.find_elements(By.TAG_NAME, 'p')
    
    # Concatenar los textos de los p en un solo comentario
    comentario_texto = ' '.join([p.text for p in parrafos])
    
    # Crear un objeto para el comentario
    comentario_objeto = {
        "comment": comentario_texto
    }
    
    # Agregar el objeto a la lista
    comentarios.append(comentario_objeto)
    
print("===============================================")
print("Array de objetos con los comentarios:")
print(comentarios)
print("===============================================")

driver.quit()

# Descargar los recursos de NLTK si no se tienen
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Inicializar el lematizador y la lista de stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('spanish'))

# Función de preprocesamiento
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^a-záéíóú\s]', '', text)
    # Tokenización
    words = nltk.word_tokenize(text)
    # Eliminar stopwords y lematizar
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Unir las palabras nuevamente en un solo string
    return ' '.join(words)

# Aplicar el preprocesamiento a todos los comentarios
comentarios_preprocesados = [{"comment": preprocess_text(comentario["comment"])} for comentario in comentarios]

print("===============================================")
print("Array de objetos con los comentarios preprocesados:")
print(comentarios_preprocesados)
print("===============================================")

# Crear el vectorizador TF-IDF
vectorizer = TfidfVectorizer()

# Obtener la matriz de características
X = vectorizer.fit_transform([comentario["comment"] for comentario in comentarios_preprocesados])

# Convertir la matriz en un DataFrame para visualizarla
df_vectorizado = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Mostrar el dataset vectorizado
print("===============================================")
print("Dataset vectorizado:")
print(df_vectorizado.head())
print("===============================================")

# Función para calcular el Silhouette Score de manera segura
def calculate_silhouette_score(X, labels):
    if len(set(labels)) > 1:
        return silhouette_score(X, labels)
    else:
        return "Silhouette Score no válido con un único cluster"

# Evaluar Agglomerative Clustering
model_agg = AgglomerativeClustering(n_clusters=3)
labels_agg = model_agg.fit_predict(df_vectorizado)
score_agg = calculate_silhouette_score(df_vectorizado, labels_agg)

print("===============================================")
print(f"Silhouette Score (Agglomerative Clustering): {score_agg}")
print("===============================================")

# Evaluar DBSCAN
model_dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = model_dbscan.fit_predict(df_vectorizado)
score_dbscan = calculate_silhouette_score(df_vectorizado, labels_dbscan)

print("===============================================")
print(f"Silhouette Score (DBSCAN): {score_dbscan}")
print("===============================================")

# Evaluar Gaussian Mixture Models
model_gmm = GaussianMixture(n_components=3)
labels_gmm = model_gmm.fit_predict(df_vectorizado)
score_gmm = calculate_silhouette_score(df_vectorizado, labels_gmm)

print("===============================================")
print(f"Silhouette Score (GMM): {score_gmm}")
print("===============================================")
