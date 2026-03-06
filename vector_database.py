import os
import faiss
import numpy as np
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import openai
import fitz

from pdf2image import convert_from_path
import pytesseract
from tempfile import TemporaryDirectory
from PIL import Image
from nltk.corpus import stopwords
from string import punctuation
import re, json

# some stuff legacy

openai_cloud = openai
openai_cloud.api_key = 'OPENAIKEY'  # Ersetze durch deinen tatsächlichen API-Schlüssel

class VectorDatabase:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L12-v2', output_dir='/output_directory/', num_topics=10, min_words=3, max_num_ratio=0.5, use_chatgpt = False):
        if use_chatgpt == True:
            self.use_chatgpt = True
            self.doc_path = output_dir
            self.model = model_name
            self.embeddings = []
            self.page_metadata = []
            self.index = None
            self.sentences = []
            self.output_dir=output_dir
            self.min_words = min_words  # Minimum number of words a sentence should have to be included
            self.max_num_ratio = max_num_ratio  # Maximum ratio of numeric characters to total characters
        
        else:  
            cache_dir = "/media/raidvol/klauth/ExpliCare/full_models/LLMs"
            self.use_chatgpt = False
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_name = model_name
            self.index = None
            self.page_metadata = []
            self.embeddings = []
            self.sentences = []
            self.topics = []
            self.output_dir = output_dir
            self.num_topics = num_topics
            self.min_words = min_words  # Minimum number of words a sentence should have to be included
            self.max_num_ratio = max_num_ratio  # Maximum ratio of numeric characters to total characters

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def print_attributes(self):
        for attr, value in self.__dict__.items():
            if value:  # Only print if the attribute is not empty
                print(f"{attr}: {value}")

    def encode(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings

    def is_valid_sentence(self, sentence):
        # Exclude short sentences
        if len(sentence.split()) < self.min_words:
            return False
        # Exclude sentences with a high ratio of numeric characters
        num_chars = sum(c.isdigit() for c in sentence)
        if num_chars / len(sentence) > self.max_num_ratio:
            return False
        return True
    
    
    def read_pdfs_by_bookmarks_from_folder2(self, folder_path):
        pdf_texts = {}

        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)

                with fitz.open(file_path) as pdf_file:
                    bookmark_texts = {}
                    toc = pdf_file.get_toc()  # Hole das Inhaltsverzeichnis (Bookmarks)

                    if toc:
                        # Durchlaufe die Bookmarks
                        for i, item in enumerate(toc):
                            level, title, page_num = item
                            bookmark_content = []
                            
                            # Bestimme die Start- und Endseiten für den aktuellen Abschnitt
                            start_page = page_num - 1
                            end_page = toc[i + 1][2] - 1 if i + 1 < len(toc) else len(pdf_file)

                            # Extrahiere den Text von der Start- bis zur Endseite
                            for page in range(start_page, end_page):
                                text = pdf_file.load_page(page).get_text("text")
                                bookmark_content.append(text)
                            
                            # Füge den gesamten Text des aktuellen Abschnitts hinzu
                            bookmark_texts[title] = "\n".join(bookmark_content)
                    else:
                        # Falls keine Bookmarks vorhanden sind, speichere den gesamten Text seitenweise
                        full_text = ""
                        for page in pdf_file:
                            full_text += page.get_text("text") + "\n"
                        bookmark_texts["No Bookmarks"] = full_text.strip()

                    pdf_texts[filename] = bookmark_texts

        return pdf_texts

    def read_pdfs_by_bookmarks_from_folder(self, folder_path):
        pdf_texts = {}
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)

                with fitz.open(file_path) as pdf_file:
                    print(pdf_file)
                    #page_texts={}
                    fulltxt = ""
                    for page_num, page in enumerate(pdf_file, start=1):
                        page_text = page.get_text("text")
                        fulltxt += page_text + "\n"
                        #page_texts[f"Seite{page_num}"] = page_text
                    pdf_texts[filename] = fulltxt

        return pdf_texts#pdf_texts


    def generate_embedding(self, text: str):
        """
        Generate an embedding for the given text using OpenAI's API.

        Parameters:
            text (str): The text to generate an embedding for.

        Returns:
            list: The embedding vector.
        """
        try:
            print(f"Anfrage an GPT Embedder mit {len(text)} char")
            response = openai_cloud.Embedding.create(
                input=text,
                model=self.model
            )
            embedding = response['data'][0]['embedding']
            print(f">>returned {len(embedding)} embeddings<<")
            return np.array(embedding).astype('float32')  # Return as numpy array
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def build_index(self, token_dir: str = "./results/" + "vector_db/" + "tokenized_sections/", evalu = False, useIp = False):

        self.page_metadata = []
        self.embeddings = []
        self.sentences = []
        all_sections = {}
        counter = 0
        
        print(f"EVALUATING AND USING ANSWERS.PDF = {evalu}")
        for filename in os.listdir(token_dir):
            if evalu == False and filename.endswith(".json") and filename != "answer_sections.json" and filename != "all_sections.json":
                filepath = os.path.join(token_dir, filename)
                print(filename)
                with open(filepath, 'r', encoding='utf-8') as json_file:
                    values = []
                    sections = json.load(json_file)
                           
                    # for entry_key, value in entry.items():
                    #     pass    
                    # if(value == "Übersetzung" or value == "Übersetzung nicht gefunden"): 
                    #     print("AAAAAAAAAAAAAAAAAAAAA")
                    #     # continue
                    # if(value in values): 
                    #         print(value + " ist doppelt. Überspringe Eintrag. \n")
                    #         # continue
                    # else:
                    #         values.append(value)
                    # if(entry_key in sections.keys()):
                    #         counter += 1
                    #         entry = {entry_key + "_" + str(counter): value}
                    #         print(entry)
                            
                    all_sections[filename] = sections
            elif evalu == True and filename.endswith(".json"):
                filepath = os.path.join(token_dir, filename)
                print(filename)
                with open(filepath, 'r', encoding='utf-8') as json_file:
                    sections = json.load(json_file)
                    # if(filename.startswith("translations")):
                    #     for entry in sections:
                    #         if isinstance(entry, dict):
                    #             for entry_key, value in entry.items():
                    #                 if(value == "Übersetzung" or value == "Übersetzung nicht gefunden"): continue
                    #                 if(value in values): 
                    #                     print(value + " ist doppelt. Überspringe Eintrag. \n")
                    #                     continue
                    #                 else:
                    #                     values.append(value)
                    #                 if(entry_key in tmp_sections.keys()):
                    #                     counter += 1
                    #                     entry = {entry_key + "_" + str(counter): value}                           
                    #             tmp_sections.update(entry)
                    #     all_sections[filename] = tmp_sections           
                    # else:
                    all_sections[filename] = sections
            # for k, v in all_sections.items():
            #     for k1, v1 in v.items():
                    
            #         print(k1 + "::" + v1)
            #         print(k)
            # exit(666)
        
        
        if(evalu== False):
            with open(os.path.join(token_dir, "all_sections.json"), 'w', encoding='utf-8') as f:
                json.dump(all_sections, f, indent=4)

        print("Loaded All files!")
        
        alr_content = []
        dbl_content = []

        for doc_name, sections in tqdm(all_sections.items(), desc="Building Index"):
            for section_title, content in tqdm(sections.items(), desc=f"Processing {doc_name}", leave=False):
                paragraphs = content
                if len(paragraphs) > 0:
                    
                    if paragraphs in alr_content:
                        print(f"!!! {paragraphs} is already in content")
                        dbl_content.append(paragraphs)
                        continue
                    else:
                        alr_content.append(paragraphs)

                        
                    self.page_metadata.append((doc_name, section_title))

                    self.sentences.append(paragraphs)

                    par_content = paragraphs 
                                      
                    if self.use_chatgpt:
                        paragraph_embeddings = self.generate_embedding(par_content)
                    else:
                        try:
                            paragraph_embeddings = self.encode(par_content)
                        except Exception as e:
                            print(f"Error: {e}")
                            print("Occurred in during processing of file:\n" + doc_name + ": " + section_title + "\n")
                    self.embeddings.append(paragraph_embeddings)
        
        if self.use_chatgpt and self.embeddings:
            print("Vectorizing with ChatGPT...")
            #self.save_embedding_to_pickle(self.embeddings, os.path.join(self.output_dir, "embeddings.pkl"))
            self.embeddings = np.vstack(self.embeddings)
            print(f"Generated {len(self.embeddings)} embeddings.")
            print(self.embeddings.shape)
            dimension = self.embeddings.shape[1]
            
            if useIp:
                normalize = faiss.NormalizationTransform(dimension)
                index = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexPreTransform(normalize, index)
            else:
                self.index = faiss.IndexFlatL2(dimension)
            
            self.index.add(self.embeddings)

        if self.use_chatgpt == False:
            print("Vectorizing with " + self.model_name + " ...")
            self.embeddings = np.vstack(self.embeddings)

            if useIp:
                normalize = faiss.NormalizationTransform(self.embeddings.shape[1])
                index = faiss.IndexFlatIP(self.embeddings.shape[1])
                self.index = faiss.IndexPreTransform(normalize, index)
            else:
                self.index = faiss.IndexFlatL2(self.embeddings.shape[1])

            self.index.add(self.embeddings)

    def perform_clustering(self):
        kmeans = KMeans(n_clusters=self.num_topics, random_state=0, n_init=10, max_iter=300)
        self.topics = kmeans.fit_predict(self.embeddings)
        # print(f"Clustering into {self.num_topics} topics completed.")

    def save(self, output_dir = None, index_filename='vector_index.faiss', meta_filename='page_metadata.pkl', embed_filename='embeddings.npy', sentences_filename='sentences.pkl', topics_filename='topics.pkl'):
        if output_dir != None:
            self.output_dir = output_dir
        if(not os.path.exists(self.output_dir)):
            os.makedirs(self.output_dir)
        index_path = os.path.join(self.output_dir, index_filename)
        meta_path = os.path.join(self.output_dir, meta_filename)
        embed_path = os.path.join(self.output_dir, embed_filename)
        sentences_path = os.path.join(self.output_dir, sentences_filename)
        if self.use_chatgpt == False:
            topics_path = os.path.join(self.output_dir, topics_filename)

        if self.index is not None:
            faiss.write_index(self.index, index_path)
        else:
            print("Index is not built. Nothing to save.")

        with open(meta_path, 'wb') as f:
            pickle.dump(self.page_metadata, f)
            
        np.save(embed_path, self.embeddings)

        with open(sentences_path, 'wb') as f:
            pickle.dump(self.sentences, f)
        if self.use_chatgpt == False:
            with open(topics_path, 'wb') as f:
                pickle.dump(self.topics, f)

        print(f"Saved index to {index_path}")
        print(f"Saved page metadata to {meta_path}")
        print(f"Saved embeddings to {embed_path}")
        print(f"Saved sentences to {sentences_path}")
        if self.use_chatgpt == False:
            print(f"Saved topics to {topics_path}")
        print("Total number of sentences in the database:", len(self.page_metadata))

    def load(self, output_dir=None, index_filename='vector_index.faiss', meta_filename='page_metadata.pkl', embed_filename='embeddings.npy', sentences_filename='sentences.pkl', topics_filename='topics.pkl'):
        if output_dir is None:
            output_dir = self.output_dir

        index_path = os.path.join(output_dir, index_filename)
        meta_path = os.path.join(output_dir, meta_filename)
        embed_path = os.path.join(output_dir, embed_filename)
        sentences_path = os.path.join(output_dir, sentences_filename)
        if self.use_chatgpt == False:
            topics_path = os.path.join(output_dir, topics_filename)

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")

        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                self.page_metadata = pickle.load(f)
        else:
            raise FileNotFoundError(f"Page metadata file not found: {meta_path}")

        if os.path.exists(embed_path):
            self.embeddings = np.load(embed_path)
        else:
            raise FileNotFoundError(f"Embeddings file not found: {embed_path}")

        if os.path.exists(sentences_path):
            with open(sentences_path, 'rb') as f:
                self.sentences = pickle.load(f)
        else:
            raise FileNotFoundError(f"Sentences file not found: {sentences_path}")
        if self.use_chatgpt == False:
            if os.path.exists(topics_path):
                with open(topics_path, 'rb') as f:
                    self.topics = pickle.load(f)
            else:
                raise FileNotFoundError(f"Topics file not found: {topics_path}")

        print(f"Loaded index from {index_path}")
        print(f"Loaded page metadata from {meta_path}")
        print(f"Loaded embeddings from {embed_path}")
        print(f"Loaded sentences from {sentences_path}")
        if self.use_chatgpt == False:
            print(f"Loaded topics from {topics_path}")
        print("Total number of sentences in the database:", len(self.page_metadata))

    def view_database(self):
        if not self.index:
            print("No index has been built yet.")
            return

        for (doc_name, section_title), embed in zip(self.page_metadata, self.embeddings): # , topic         , self.topics
            print(f"Document: {doc_name}, Page: {section_title}")
            #print(f"Sentence embedding: {embed}...")  # Show only the first 10 dimensions of the embedding
            print(f"Sentence text: {self.sentences[self.page_metadata.index((doc_name, section_title))]}")
            # print(f"Topic: {topic}")
            print("-" * 40)  # Separator for clarity

        print("Total number of sentences in the database:", len(self.page_metadata))
        print("-" * 40)

    def query(self, text, k=5):
        if self.use_chatgpt:
            query_vector = self.generate_embedding(text)
        else:
            query_vector = self.encode([text])[0]
        distances, indices = self.index.search(np.array([query_vector]), k)
        results = [(self.page_metadata[i], distances[0][j], self.sentences[i]) for j, i in enumerate(indices[0])] # , self.topics[i]
        return results
    
    # def _preprocess_sentence(self, sentence: str):
    #    sentence = re.sub('[^0-9a-zA-ZäÄöÖüÜß]+', ' ', sentence)
    #    sentence = sentence.lower().lstrip().rstrip()
    #    sentence = re.sub('\s+', ' ', sentence)
    #    res = []
           
    #    for word in sentence.split(" "):
    #        word = word.strip()
    #        word = "".join([x for x in word if x not in punctuation])
    #        if word not in stopwords.words('german') and word not in punctuation:
    #            res.append(word)
       
    #    return sentence
    
    
    # def load_documents_from_folder(self, doc_path):
    #     page_contents = {}
    #     doc_files = os.listdir(doc_path)
    #     doc_dir = [name[:-4] if name.lower().endswith('.pdf') else name for name in doc_files]
    #     print(f"Processing {len(doc_dir)} Documents!")
    #     for doc in tqdm(doc_dir, desc="Processing Documents"):
    #             file_path = f"{doc_path}{doc}.pdf"
    #             # print("Procssing Document: " + doc)
    #             pages = []
    #             image_files = []

    #             pdf_pages = convert_from_path(file_path,500)
    #             with TemporaryDirectory() as tempdir:
    #                 for page_enum,page in enumerate(pdf_pages):
    #                     fname = f"{tempdir}\page_{page_enum:03}.jpg"
    #                     page.save(fname,"JPEG")
    #                     image_files.append(fname)
    #                 for image_file in image_files:
    #                     text = str(((pytesseract.image_to_string(Image.open(image_file),lang="deu"))))
    #                     text = text.replace("-\n","")
    #                     # print(text)
    #                     pages.append(text)
    #                 # print(pages[0])
                
    #             text_de = pages
    #             #text_de = [doc["description"]]
    #             text_de = [x.lower().replace("\n"," ") for x in text_de]
    #             res_text_de = []
    #             for i,sentence in enumerate(text_de):
    #                 sent =  self._preprocess_sentence(sentence)
    #                 if len(sent.strip())<self.min_words: # sentence too short
    #                     continue
    #                 text_de[i] = self._preprocess_sentence(sentence)
    #             page_contents[doc] = text_de 
    #     # with open("corpus_en.json","w",encoding="utf-8") as df:
    #     #         json.dump(page_contents,df,ensure_ascii=False)
    #     print("Documents processed")
    #     return page_contents
                
    # def load_documents_from_folder(self, folder_path):
    #     page_contents = {}
    #     for filename in os.listdir(folder_path):
    #         file_path = os.path.join(folder_path, filename)
    #         if os.path.isfile(file_path):
    #             if filename.endswith('.txt'):
    #                 page_contents[filename] = self.read_text_file_by_pages(file_path)
    #             elif filename.endswith('.pdf'):
    #                 page_contents[filename] = self.read_pdf_file_by_pages(file_path)
    #     return page_contents

    def read_text_file_by_pages(self, file_path, page_size=500):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        pages = [content[i:i + page_size] for i in range(0, len(content), page_size)]
        return pages

    def del_vectorizer(self):
        del self.model # Um speicher der GPU wieder freizugeben. Model wird nicht mehr benötigt
        torch.cuda.empty_cache()
        print("Cache Empty!")

    def read_pdf_file_by_pages(self, file_path):
        pages = []
        import fitz  # PyMuPDF
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text = page.get_text()
                pages.append(text)
        return pages


    def generate_chatGPT_embedding(text: str, model="text-embedding-ada-002"):
        """
        Generate embeddings using the OpenAI API.

        Parameters:
            text (str): The input text for generating the embedding.
            model (str): The embedding model to use. Default is 'text-embedding-ada-002'.

        Returns:
            list: The embedding vector.
        """
        try:
            print(f">>Anfrage an GPT Embedder mit {len(text)} char<<")
            response = openai_cloud.Embedding.create(
                input=text,
                model=model
            )
            embedding = response['data'][0]['embedding']
            print(f">>returned {len(embedding)} embeddings<<")
            return embedding
        except Exception as e:
            print(f"Error: {e}")
            return None

    def save_embedding_to_pickle(embedding, filename="embedding.pkl"):
        """
        Save the embedding to a binary file using pickle.

        Parameters:
            embedding (list): The embedding vector to save.
            filename (str): The name of the file to save to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(embedding, f)
        print(f"Embedding saved to {filename}")

    
