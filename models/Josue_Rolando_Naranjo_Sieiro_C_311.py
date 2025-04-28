
NAME = "Josue Rolando Naranjo Sieiro"
GROUP = "311"
CAREER = "Ciencia de la Computación"
MODEL = "Modelo de Semántica Latente (Latent Semantic Indexing)"

"""
INFORMACIÓN EXTRA:

Fuente bibliográfica:
- Information Retrieval WS 17/18, Lecture 10: Latent Semantic Indexing -https://www.youtube.com/watch?v=CwBn0voJDaw
- Latent Semantic Indexing | Explained with Examples | Georgia Tech CSE6242 - https://www.youtube.com/watch?v=M1duqgg8-IM (joya)
- Scikit-learn documentation. (2024). TF-IDF feature extraction.  https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- Scikit-learn documentation. (2024). Latent Semantic Analysis using TruncatedSVD.  https://scikit-learn.org/stable/modules/decomposition.html#latent-semantic-analysis
- Wikipedia contributors. (2024). Latent Semantic Analysis. https://en.wikipedia.org/wiki/Latent_semantic_analysis
- Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K. & Harshman, R. (1990). Indexing by Latent Semantic Analysis. *Journal of the American Society for Information Science*, 41(6): 391–407. URL: https://www.cs.csustan.edu/~mmartin/LDS/Deerwester-et-al.pdf :contentReference[oaicite:0]{index=0}
- Dumais, S. T., Furnas, G. W., Landauer, T. K., Deerwester, S. C. & Harshman, R. (1988). Using latent semantic analysis to improve access to textual information. *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems*. URL: https://www.researchgate.net/publication/2462489_Using_Latent_Semantic_Analysis_To_Improve_Access_To_Textual_Information :contentReference[oaicite:1]{index=1}
- Berry, M. W., Dumais, S. T. & O’Brien, G. W. (1995). Using linear algebra for intelligent information retrieval. *SIAM Review*, 37(4): 573–595. :contentReference[oaicite:2]{index=2}
- Manning, C. D., Raghavan, P. & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. URL: https://nlp.stanford.edu/IR-book/information-retrieval-book.html :contentReference[oaicite:3]{index=3}
- Baeza-Yates, R. & Ribeiro-Neto, B. (2011). *Modern Information Retrieval: The Concepts and Technology Behind Search* (2ª ed.). Addison-Wesley. URL: https://www.amazon.com/Modern-Information-Retrieval-Concepts-Technology/dp/0321416910 :contentReference[oaicite:4]{index=4}

Mejora implementada:
- Se configuró el vectorizador TF-IDF con `lowercase=True`, `stop_words='english'` y `max_df=0.8` para filtrar términos triviales y estandarizar el texto antes de la descomposición SVD.  
  Beneficio esperado: reduce el ruido en la matriz TF-IDF y mejora la calidad del espacio latente, incrementando la precisión y consistencia de las similitudes.

Definición del modelo:
Q: Vector latente de la consulta, obtenido al transformar el TF-IDF de la query y proyectarlo con TruncatedSVD.  
D: Vector latente de cada documento, resultado de aplicar TruncatedSVD a la matriz TF-IDF completa.  
F: 
\[
\text{sim}(Q,D) = \frac{Q_{\text{latent}} \cdot D_{\text{latent}}}{\|Q_{\text{latent}}\|\,\|D_{\text{latent}}\|}
\]
R: Se ordenan los documentos de mayor a menor similitud y se devuelven los top_k.

¿Dependencia entre los términos?  
Sí. LSI incorpora dependencia al capturar patrones de co-ocurrencia en la descomposición SVD, revelando relaciones semánticas entre términos.

Correspondencia parcial documento-consulta?  
Sí. Permite recuperar documentos que no comparten palabras literales con la consulta, pero sí conceptos semánticos afines.

Ranking?  
Sí. Los documentos se clasifican por similitud de coseno en el espacio latente, priorizando los más cercanos conceptualmente.
"""


import ir_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import random
from typing import Dict, List, Tuple
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# singular value decomposition (Siempre se me olvida xD)

class InformationRetrievalModel:
    def __init__(self):
        """
        Inicializa el modelo de recuperación de información.
        """
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',      
            max_df=0.8,                # ignora terminos demasiado frecuentes
        )
        self.tfidf_matrix = None
        self.documents = []
        self.doc_ids = []
        self.dataset = None
        self.queries = {}
    
    def fit(self, dataset_name: str):
        """
        Carga y procesa un dataset de ir_datasets, incluyendo todas sus queries.
        
        Args:
            dataset_name (str): Nombre del dataset en ir_datasets (ej: 'cranfield')
        """
        # Cargar dataset
        self.dataset = ir_datasets.load(dataset_name)
        
        if not hasattr(self.dataset, 'queries_iter'):
            raise ValueError("Este dataset no tiene queries definidas")
        
        self.documents = []
        self.doc_ids = []
        
        for doc in self.dataset.docs_iter():
            self.doc_ids.append(doc.doc_id)
            self.documents.append(doc.text)
            
        # Vectorizar documentos en matriz TF-IDF (filas == docs, columnas == terminos)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        
        self.queries = {q.query_id: q.text for q in self.dataset.queries_iter()}
        
        k = 100 # Numero de dimensiones latentes
        
        # from sklearn.decomposition import TruncatedSVD
        
        # Crear el modelo SVD truncado 
        self.svd = TruncatedSVD(n_components=k, random_state=42)
        
        # Ahora se ajusta SVD a la matriz TF_IDF de documentos 
        # y se obtienen representaciones latentes 
        self.doc_latent_matrix = self.svd.fit_transform(self.tfidf_matrix) 
    
    def predict(self, top_k: int) -> Dict[str, Dict[str, List[str]]]:
        """
        Realiza búsquedas para TODAS las queries del dataset automáticamente.
        
        Args:
            top_k (int): Número máximo de documentos a devolver por query.
            threshold (float): Umbral de similitud mínimo para considerar un match.
            
        Returns:
            dict: Diccionario con estructura {
                query_id: {
                    'text': query_text,
                    'results': [(doc_id, score), ...]
                }
            }
        """
        results = {}
        
        for qid, query_text in self.queries.items():
            # Vectorizar la consulta con el TF_IDF existente (sin volver a fit!)
            query_vec = self.vectorizer.transform([query_text])
            
            # Proyectar la consulta al espacio latente usando el SVD entrenado
            query_latent = self.svd.transform(query_vec)
            
            # Calcular la similitud de coseno entre la consulta y los docs
            # from sklearn.metrics.pairwise import cosine_similarity
            sim_scores = cosine_similarity(query_latent, self.doc_latent_matrix)[0]
            
            # Indices de docs ordenados por > similitud
            top_index = sim_scores.argsort()[::-1][:top_k]
            
            # Resultados : doc_id y puntuaje de similitud
            ranked_docs = [(self.doc_ids[i], float(sim_scores[i])) for i in top_index]
            retrieved_ids = [doc_id for doc_id, _ in ranked_docs]
            results[qid] = {
                'text' : query_text,
                'results' : retrieved_ids
            } 
            
        return results
    
    def evaluate(self, top_k: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Evalúa los resultados para TODAS las queries comparando con los qrels oficiales.
        
        Args:
            top_k (int): Número máximo de documentos a considerar por query.
            
        Returns:
            dict: Métricas de evaluación por query y métricas agregadas.
        """
        if not hasattr(self.dataset, 'qrels_iter'):
            raise ValueError("Este dataset no tiene relevancias definidas (qrels)")
        
        predictions = self.predict(top_k=top_k)
        
        qrels = {}
        for qrel in self.dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        
        result = {}
        
        for qid, data in predictions.items():
            if qid not in qrels:
                continue
                
            relevant_docs = set(doc_id for doc_id, rel in qrels[qid].items() if rel > 0)
            retrieved_docs = set(data['results'])
            relevant_retrieved = relevant_docs & retrieved_docs
            
            result[qid] = {
                'all_relevant': relevant_docs,
                'all_retrieved': retrieved_docs,
                'relevant_retrieved': relevant_retrieved
            }
        
        return result
