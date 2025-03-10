import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone as PineconeClient
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import logging
from pathlib import Path

# Configura il logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carica le variabili d'ambiente da .env
load_dotenv()

# Recupera le chiavi API dalle variabili d'ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Configurazione specifica per l'indice test-crocerossa
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "test-crocerossa")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "documenti_crocerossa")
PINECONE_DIMENSIONS = 1024  # Corrispondente alla dimensione utilizzata
PINECONE_HOST = os.getenv("PINECONE_HOST", "https://test-crocerossa-vfojmmw.svc.aped-4627-b74a.pinecone.io")

# Modelli predefiniti
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 5

# Dizionario per memorizzare le conversazioni attive
active_conversations = {}

# Definizione dei modelli Pydantic per gli input e output API
class Query(BaseModel):
    query: str
    session_id: Optional[str] = None
    
class SessionReset(BaseModel):
    session_id: str
    
class ElevenLabsWebhook(BaseModel):
    text: str = ""
    conversation_id: Optional[str] = None

class RAGSystem:
    """
    Sistema RAG universale che può essere utilizzato tramite API o webhook.
    """
    
    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        index_name: str = PINECONE_INDEX_NAME,
        namespace: str = PINECONE_NAMESPACE,
        top_k: int = DEFAULT_TOP_K
    ):
        """
        Inizializza il sistema RAG.
        
        Args:
            embedding_model: Il modello di embedding da utilizzare
            llm_model: Il modello LLM da utilizzare per la generazione
            temperature: La temperatura per la generazione
            index_name: Il nome dell'indice Pinecone
            namespace: Il namespace nell'indice Pinecone
            top_k: Numero di documenti da recuperare
        """
        # Verifica delle chiavi API
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY non trovata. Aggiungi questa chiave al file .env")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY non trovata. Aggiungi questa chiave al file .env")
        
        # Inizializza i modelli e le configurazioni
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.index_name = index_name
        self.namespace = namespace
        self.top_k = top_k
        
        # Inizializza il client Pinecone
        try:
            self.pc = PineconeClient(api_key=PINECONE_API_KEY)
            self.index = self.pc.Index(index_name)
            logger.info(f"Connessione a Pinecone stabilita. Indice: {index_name}")
        except Exception as e:
            logger.error(f"Errore nella connessione a Pinecone: {str(e)}")
            raise
        
        # Inizializza gli embeddings
        try:
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=OPENAI_API_KEY
            )
            logger.info(f"Modello di embedding inizializzato: {embedding_model}")
        except Exception as e:
            logger.error(f"Errore nell'inizializzazione del modello di embedding: {str(e)}")
            raise
        
        # Inizializza il modello LLM
        try:
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=temperature,
                openai_api_key=OPENAI_API_KEY
            )
            logger.info(f"Modello LLM inizializzato: {llm_model}")
        except Exception as e:
            logger.error(f"Errore nell'inizializzazione del modello LLM: {str(e)}")
            raise
        
        # Template per la generazione delle risposte
        self.qa_template = """
        Sei un assistente virtuale esperto della Croce Rossa. Rispondi alle domande dell'utente 
        basandoti esclusivamente sulle informazioni fornite nel contesto. Se le informazioni non sono 
        sufficienti per rispondere in modo completo, indica chiaramente cosa non puoi rispondere 
        sulla base delle informazioni disponibili. Non inventare informazioni.
        
        Contesto: {context}
        
        Storico della conversazione:
        {chat_history}
        
        Domanda: {question}
        
        Risposta:
        """
    
    def get_memory_for_session(self, session_id: str) -> ConversationBufferMemory:
        """
        Ottiene o crea una memoria per la sessione specificata.
        
        Args:
            session_id: L'ID della sessione
            
        Returns:
            La memoria della conversazione per la sessione
        """
        if session_id not in active_conversations:
            active_conversations[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        
        return active_conversations[session_id]
    
    def resize_embedding(self, embedding: List[float], target_dim: int = PINECONE_DIMENSIONS) -> List[float]:
        """
        Ridimensiona un vettore di embedding alla dimensione target.
        Uguale alla funzione nel file di caricamento per mantenere la coerenza.
        
        Args:
            embedding: Il vettore di embedding da ridimensionare
            target_dim: La dimensione target
            
        Returns:
            Il vettore ridimensionato
        """
        current_dim = len(embedding)
        
        if current_dim == target_dim:
            return embedding
        
        if current_dim < target_dim:
            # Padding con zeri se il vettore è più piccolo del target
            return embedding + [0.0] * (target_dim - current_dim)
        else:
            # Restituisce le prime 'target_dim' componenti
            resized = embedding[:target_dim]
            
            # Normalizzazione per mantenere la lunghezza unitaria del vettore
            magnitude = sum(x*x for x in resized) ** 0.5
            if magnitude > 0:
                normalized = [x/magnitude for x in resized]
                return normalized
            
            return resized
    
    def query_pinecone(self, query_embedding: List[float]) -> List[Document]:
        """
        Esegue una query su Pinecone e restituisce i documenti più rilevanti.
        
        Args:
            query_embedding: L'embedding della query
            
        Returns:
            Una lista di Document da LangChain
        """
        try:
            # Ridimensiona l'embedding se necessario
            resized_embedding = self.resize_embedding(query_embedding)
            
            # Esegui la query su Pinecone
            results = self.index.query(
                vector=resized_embedding,
                top_k=self.top_k,
                namespace=self.namespace,
                include_metadata=True
            )
            
            # Converti i risultati in Document di LangChain
            documents = []
            for match in results.matches:
                doc = Document(
                    page_content=match.metadata.get("text", "Contenuto non disponibile"),
                    metadata={
                        "score": match.score,
                        "file_name": match.metadata.get("file_name", ""),
                        "source": match.metadata.get("source", "")
                    }
                )
                documents.append(doc)
            
            logger.info(f"Query Pinecone completata. Documenti recuperati: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"Errore nella query su Pinecone: {str(e)}")
            return []
    
    def retrieve_relevant_documents(self, query: str) -> List[Document]:
        """
        Recupera i documenti più rilevanti per la query.
        
        Args:
            query: La query testuale
            
        Returns:
            Una lista di Document da LangChain
        """
        try:
            # Genera l'embedding per la query
            query_embedding = self.embeddings.embed_query(query)
            
            # Recupera i documenti rilevanti
            return self.query_pinecone(query_embedding)
        except Exception as e:
            logger.error(f"Errore nel recupero dei documenti: {str(e)}")
            return []
    
    def format_retrieved_documents(self, documents: List[Document]) -> str:
        """
        Formatta i documenti recuperati in un unico contesto.
        
        Args:
            documents: Lista di documenti recuperati
            
        Returns:
            Il contesto formattato come stringa
        """
        if not documents:
            return "Nessun documento pertinente trovato."
            
        formatted_docs = []
        
        for i, doc in enumerate(documents):
            formatted_doc = f"Documento {i+1} [fonte: {doc.metadata.get('file_name', 'sconosciuta')}]:\n{doc.page_content}\n"
            formatted_docs.append(formatted_doc)
        
        return "\n".join(formatted_docs)
    
    async def answer_question(
        self, 
        question: str, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Risponde a una domanda utilizzando il sistema RAG.
        
        Args:
            question: La domanda dell'utente
            session_id: ID di sessione opzionale per tracciare la conversazione
            
        Returns:
            Un dizionario contenente la risposta e i documenti recuperati
        """
        try:
            # Se non c'è un session_id, ne crea uno
            if not session_id:
                session_id = str(uuid.uuid4())
                logger.info(f"Creato nuovo session_id: {session_id}")
            else:
                logger.info(f"Utilizzato session_id esistente: {session_id}")
            
            # Ottieni la memoria per la sessione
            memory = self.get_memory_for_session(session_id)
            
            # Recupera i documenti rilevanti
            documents = self.retrieve_relevant_documents(question)
            if not documents:
                logger.warning("Nessun documento recuperato per la query")
            
            # Formatta i documenti in un contesto
            context = self.format_retrieved_documents(documents)
            
            # Crea il prompt
            prompt = PromptTemplate(
                template=self.qa_template,
                input_variables=["context", "question", "chat_history"]
            )
            
            # Prepara l'input per la chain
            inputs = {
                "context": context,
                "question": question,
                "chat_history": memory.buffer
            }
            
            # Genera la risposta direttamente con l'LLM
            response_text = self.llm.invoke(
                prompt.format(**inputs)
            ).content
            
            # Aggiungi alla memoria
            memory.save_context(
                {"input": question},
                {"output": response_text}
            )
            
            # Prepara i metadati delle fonti
            sources = []
            for doc in documents:
                if doc.metadata:
                    sources.append(doc.metadata)
            
            # Prepara il risultato
            result = {
                "answer": response_text,
                "sources": sources,
                "session_id": session_id
            }
            
            logger.info(f"Generata risposta per session_id: {session_id}")
            return result
            
        except Exception as e:
            error_msg = f"Errore durante l'elaborazione della richiesta: {str(e)}"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "answer": "Mi dispiace, si è verificato un errore durante l'elaborazione della tua domanda.",
                "sources": [],
                "session_id": session_id
            }
    
    async def elevenlabs_webhook_handler(self, request_data: ElevenLabsWebhook) -> Dict[str, Any]:
        """
        Gestisce le richieste webhook da ElevenLabs.
        
        Args:
            request_data: I dati della richiesta webhook
            
        Returns:
            La risposta formattata per ElevenLabs
        """
        try:
            # Log della richiesta
            logger.info(f"Ricevuta richiesta webhook da ElevenLabs: {request_data}")
            
            # Estrai i dati dalla richiesta di ElevenLabs
            input_text = request_data.text
            conversation_id = request_data.conversation_id or str(uuid.uuid4())
            
            if not input_text:
                logger.warning("Ricevuta richiesta webhook senza testo")
                return {
                    "text": "Non è stata fornita alcuna domanda.",
                    "conversation_id": conversation_id
                }
            
            # Processa la query con il sistema RAG
            result = await self.answer_question(input_text, session_id=conversation_id)
            
            # Formatta la risposta per ElevenLabs
            response = {
                "text": result["answer"],
                "conversation_id": conversation_id
            }
            
            logger.info(f"Inviata risposta webhook per conversation_id: {conversation_id}")
            return response
            
        except Exception as e:
            error_msg = f"Errore nel webhook: {str(e)}"
            logger.error(error_msg)
            return {
                "text": "Mi dispiace, si è verificato un errore durante l'elaborazione della richiesta.",
                "conversation_id": conversation_id
            }
    
    def reset_conversation(self, session_id: str) -> Dict[str, Any]:
        """
        Resetta la memoria della conversazione.
        
        Args:
            session_id: ID di sessione
            
        Returns:
            Un messaggio di conferma
        """
        if session_id in active_conversations:
            active_conversations[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            logger.info(f"Conversazione resettata per session_id: {session_id}")
        else:
            logger.warning(f"Tentativo di reset per una sessione inesistente: {session_id}")
        
        return {
            "status": "success",
            "message": "Conversazione resettata con successo",
            "session_id": session_id
        }


# Funzioni di utilità per l'utilizzo del sistema RAG

def create_rag_system(
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    llm_model: str = DEFAULT_LLM_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K
) -> RAGSystem:
    """
    Crea un'istanza del sistema RAG con i parametri specificati.
    
    Args:
        embedding_model: Il modello di embedding da utilizzare
        llm_model: Il modello LLM da utilizzare
        temperature: La temperatura per la generazione
        top_k: Numero di documenti da recuperare
        
    Returns:
        Un'istanza del sistema RAG
    """
    logger.info(f"Creazione sistema RAG con embedding_model={embedding_model}, llm_model={llm_model}")
    return RAGSystem(
        embedding_model=embedding_model,
        llm_model=llm_model,
        temperature=temperature,
        top_k=top_k
    )


# Inizializza l'app FastAPI
app = FastAPI(title="RAG System API", description="API per il sistema RAG della Croce Rossa")

# Inizializza il sistema RAG (lazy initialization)
rag_system = None

def get_rag_system():
    """
    Ottiene o inizializza il sistema RAG.
    
    Returns:
        Un'istanza del sistema RAG
    """
    global rag_system
    if rag_system is None:
        rag_system = create_rag_system()
    return rag_system

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consenti tutte le origini (in produzione limitare alle origini necessarie)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configura i file statici
static_files_dir = Path("static")
if static_files_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Route principale per servire l'index.html
@app.get("/")
async def index():
    """
    Serve la pagina principale dell'applicazione.
    """
    if (static_files_dir / "index.html").exists():
        return FileResponse("static/index.html")
    return {"message": "API RAG System attiva"}

@app.post("/elevenlabs-webhook")
async def elevenlabs_webhook(data: ElevenLabsWebhook):
    """
    Endpoint webhook per ElevenLabs.
    """
    logger.info(f"Ricevuta richiesta a /elevenlabs-webhook: {data}")
    response = await get_rag_system().elevenlabs_webhook_handler(data)
    return response


@app.post("/langchain-query")
async def langchain_query(query_data: Query):
    """
    Endpoint per query dirette da LangChain.
    """
    logger.info(f"Ricevuta richiesta a /langchain-query: {query_data}")
    
    response = await get_rag_system().answer_question(query_data.query, query_data.session_id)
    
    return response


@app.post("/reset-conversation")
async def reset_conversation(session_data: SessionReset):
    """
    Endpoint per resettare una conversazione.
    """
    logger.info(f"Ricevuta richiesta a /reset-conversation: {session_data}")
    
    response = get_rag_system().reset_conversation(session_data.session_id)
    
    return response

# Endpoint di compatibilità per API transcript e contatti
@app.get("/api/transcript")
async def get_transcript():
    """
    Endpoint per recuperare la trascrizione della conversazione.
    Funzione stub per compatibilità con il frontend.
    """
    return {"transcript_html": "<p>Funzionalità transcript non ancora implementata.</p>"}

@app.get("/api/extract_contacts")
async def extract_contacts():
    """
    Endpoint per estrarre i contatti dalla conversazione.
    Funzione stub per compatibilità con il frontend.
    """
    return {"contact_info": "<p>Funzionalità di estrazione contatti non ancora implementata.</p>"}

# Funzione per l'utilizzo diretto tramite import
async def query_rag(query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Funzione per interrogare direttamente il sistema RAG.
    Utile per l'integrazione con altri sistemi.
    
    Args:
        query: La domanda dell'utente
        session_id: ID di sessione opzionale
        
    Returns:
        La risposta del sistema RAG
    """
    return await get_rag_system().answer_question(query, session_id)


if __name__ == "__main__":
    import uvicorn
    
    # Imposta la porta dinamica per Railway
    port = int(os.environ.get("PORT", 5000))
    
    # Log porta attuale
    logger.info(f"Avvio del server sulla porta {port}...")

    # Avvia il server uvicorn con il nome del file corrente
    uvicorn.run("rag:app", host="0.0.0.0", port=port, reload=True)