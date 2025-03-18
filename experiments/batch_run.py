from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from inspect import signature
from yaml import safe_load
import pandas as pd 
import tqdm
import re
import json
from pathlib import Path
import os
import faiss
import gc
from typing import Tuple

OUTPUT_DIR = Path("test_results")
today = pd.Timestamp.now().strftime("%Y-%m-%d")

class XRIFGenerator:
    def __init__(self, 
                 prompt_template: str,
                 waypoints_csv: str,
                 language_model_type = "Ollama",
                 embed_model_type = "Ollama",
                 language_model: str = "deepseek-r1:1.5b",
                 embed_model: str = "deepseek-r1:1.5b",
                 chat_kwargs: dict = {"num_predict": 500, "temperature": 0.5, "top_p": 0.5},
                 fetch_all: bool = True):
        self.waypoints_list = []
        self.kwargs = chat_kwargs
        self.prompt_template = prompt_template
        self.waypoints_csv = waypoints_csv
        self.x_coord = 0
        self.y_coord = 0
        print(f"Loading model: {language_model}...")
        
        if embed_model_type == "Ollama":
            embed = OllamaEmbeddings(model = embed_model)
        elif embed_model_type == "HuggingFace":
            embed = HuggingFaceEmbeddings(model_name = embed_model)
        else:
            raise Exception("Invalid embed_model_type")
        
        if language_model_type == "Ollama":
            self.llm = OllamaLLM(model=language_model, **self.filter_kwargs_for_OllamaLLM(self.kwargs))
            self.output_parser = StrOutputParser()
        else:
            raise Exception("Invalid language_model_type")
        

        waypoints = pd.read_csv(self.waypoints_csv)
        waypoint_cols = ['Location', 'X co-ordinate', 'Y co-ordinate', 'Floor', 'Section', 'Keywords']
        given_waypoints_cols = list(waypoints.columns)
        if given_waypoints_cols != waypoint_cols:
            raise Exception("Columns do not match, please provide a new dataset")
        
        num_rows = len(waypoints)
        print(f"Processing {num_rows} waypoints...")
        with tqdm.tqdm(total=num_rows) as pbar:
            for index, row in waypoints.iterrows():
                keywords_string = ', '.join(row['Keywords'])
                self.waypoints_list.append(Document(id = f"Waypoint_{index}", page_content=f"A waypoint called {row['Location']} exists at X-Coordinate, {row['X co-ordinate']}, and Y-Coordinate, {row['Y co-ordinate']}, apart of Section, {row['Section']}. Words associated with the waypoint, {row['Location']} are {keywords_string}."))
                pbar.update(1)
        
        index = faiss.IndexFlatL2(len(embed.embed_query("Hello World!")))

        self.vector_store = FAISS(
            embedding_function=embed,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}            
        )
        
        print("Adding waypoints to vector store...")
        self.vector_store.add_documents(self.waypoints_list)
        print("Creating retriever...")
        k = len(self.waypoints_list) if fetch_all else 5
        
        self.retriever = self.vector_store.as_retriever(
            search_type = "mmr",
            search_kwargs = {'k': k, 'fetch_k': len(self.waypoints_list)},
        )

        with open(self.prompt_template, 'r') as file:
            prompt_dict = safe_load(file)
            self.prompt = prompt_dict['prompt']
            self.prompt_name = prompt_dict['prompt_name']
        
        print("Creating prompt template...")
        self.prompt_template = PromptTemplate(template=self.prompt, input_variables=["documents", "query", "starting_location"])

        print("Creating RAG chain...")
        self.rag_chain = self.prompt_template | self.llm | self.output_parser
        
    def filter_kwargs_for_OllamaLLM(self, kwargs: dict) -> dict:
        chatollama_params = signature(OllamaLLM.__init__).parameters
        return {k: v for k, v in kwargs.items() if k in chatollama_params}
    
    def generate_xrif_with_deepseek_model(self, query: str):
        xrif_response = None
        documents = self.retriever.get_relevant_documents(query)
        doc_texts = "\\n".join([doc.page_content for doc in documents])

        llm_response = self.rag_chain.invoke({"documents": doc_texts, "query": query, "starting_location": f"X-Coordinate: {self.x_coord}, Y-Coordinate: {self.y_coord}"})

        xrif_response = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL)

        xrif_response = re.sub(r"\n", "", xrif_response)

        xrif_response = re.sub(r"\\n", "", xrif_response)

        start = xrif_response.find('{')
        end = xrif_response.rfind('}')

        if start != -1 and end != -1:
            xrif_response = xrif_response[start:end+1]
            try:
                xrif_response = json.loads(xrif_response)
            except json.JSONDecodeError as e:
                print("Error parsing JSON response")

        
        return llm_response, xrif_response
    
    def batch_test_run(self, df: pd.DataFrame, output_folder: Path):
        num_rows = len(df)
        print(f"Processing {num_rows} test prompts...")
        with tqdm.tqdm(total=num_rows) as pbar:
            for index, row in df.iterrows():
                now = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
                dump_json = {}
                error_message = 'No Error'
                note = ''
                query = row['Prompt']
                self.x_coord = row['Starting X']
                self.y_coord = row['Starting Y']
                expected_response = row['Expected Response']
                llm_response, xrif_response = self.generate_xrif_with_deepseek_model(query)

                dump_json['Data Set'] = row['Data Set']
                dump_json['Experiment ID'] = row['Experiment ID']
                dump_json['Prompt ID'] = row['Prompt ID']
                dump_json['Prompt'] = query
                dump_json['Full LLM Response'] = str(llm_response)
                dump_json['XRIF Generated'] = xrif_response if isinstance(xrif_response, dict) else str(xrif_response)
                dump_json['Starting X'] = self.x_coord
                dump_json['Starting Y'] = self.y_coord
                dump_json['Expected Response'] = expected_response
                dump_json['Expected Response Type'] = row['Expected Response Type']
                expected_response = list(expected_response)
                if xrif_response and type(xrif_response) == dict:
                    if 'actions' in xrif_response.keys():
                        actions = xrif_response['actions']
                        if len(expected_response) == len(actions):
                            if row['Expected Response Type'] == 'Nav':
                                list_of_locations = [action['input']['name'] for action in actions if (action['action'] == 'navigate') and ('input' in action) and ('name' in action['input'])]
                                set_of_locations = set(list_of_locations)
                                set_of_expected_locations = set(expected_response)
                                if set_of_locations == set_of_expected_locations:
                                    note = "All locations are present in the response"
                                    dump_json['Notes'] = note
                            for i in range(len(xrif_response['actions'])):
                                if actions[i]['action'] == 'navigate' :
                                    if type(expected_response[i]) == 'str':
                                        if 'input' in actions[i]:
                                            if 'name' in actions[i]['input'].keys():
                                                if actions[i]['input']['name'] == expected_response[i] or (type(expected_response[i]) == 'list' and actions[i]['input']['name'] in expected_response[i]):
                                                    continue
                                                else:
                                                    error_message = f"Expected location name: {actions[i]['name']} does not match with the provided location name: {expected_response[i]}"
                                                    dump_json['Error Message'] = error_message
                                                    break
                                            else:
                                                error_message = f"Response Error Missing field: 'name' in action object {i}"
                                                dump_json['Error Message'] = error_message
                                                break
                                        else:
                                            error_message = f"Response Error Missing field: 'input' in action object {i}"
                                            dump_json['Error Message'] = error_message
                                            break
                                    else:
                                        error_message = f"Expected Response Error: Expected Response Type is not 'Nav' for action object {i}"
                                        dump_json['Error Message'] = error_message
                                        break
                                elif actions[i]['action'] == 'wait':
                                    if type(expected_response[i]) == 'tuple':
                                        if 'input' in actions[i]:
                                                if actions[i]['action'] != expected_response[i][0]:
                                                    error_message = f"Expected action at action object {i} : {expected_response[i][0]}  does not match with the provided action: {actions[i]['action']}"
                                                    dump_json['Error Message'] = error_message
                                                    break
                                                else:
                                                    if (int(actions[i]['input']))/60 == expected_response[i][1]:
                                                        error_message = f"Expected wait time: {actions[i]['input']} does not match with the provided wait time: {expected_response[i][1]}"
                                                        dump_json['Error Message'] = error_message
                                                        break
                                        else:
                                            error_message = f"Response Error Missing field: 'input' in action object {i}"
                                            dump_json['Error Message'] = error_message
                                            break
                                    else:
                                        error_message = f"Expected Response Error: Expected Response Type is not 'wait' for action object {i}"
                                        dump_json['Error Message'] = error_message
                                        break
                                elif actions[i]['action'] == 'speak':
                                    if type(expected_response[i]) == 'tuple':
                                        if ('action' in actions[i]) and ('input' in actions[i]):
                                            if actions[i]['action'] == expected_response[i][0] and (actions[i]['input'] == expected_response[i][1] or (type(expected_response[i]) == 'list' and actions[i]['input'] in expected_response[i])):
                                                continue
                                            else:
                                                error_message = f"Expected speak message: {expected_response[i]} does not match with the generated speak message: {actions[i]['input']} or Expected action: {expected_response[i][0]} does not match with the generated action: {actions[i]['action']}"
                                                dump_json['Error Message'] = error_message
                                                break
                                        else:
                                            error_message = f"Response Error Missing field: 'input' or 'action' in action object {i}"
                                            dump_json['Error Message'] = error_message
                                            break
                                    else:
                                        error_message = f"Expected Response Error: Expected Response Type is not correct for action object {i}"
                                        dump_json['Error Message'] = error_message
                                        break
                else:
                    error_message = "XRIF response is Invalid"
                    dump_json['Error Message'] = error_message
                
                if not os.path.exists(OUTPUT_DIR / output_folder):
                    os.makedirs(OUTPUT_DIR / output_folder)

                with open(OUTPUT_DIR / output_folder / f"Prompt_{row['Prompt ID']}_{now}.json", 'w') as file:
                    json.dump(dump_json, file, indent=4)
                pbar.update(1)

        return OUTPUT_DIR / output_folder

def process_test_dataset(dataset: pd.DataFrame):
    dataset['Expected Response'] = dataset['Expected Response'].to_list()
    dataset.dropna(subset=['Prompt', 'Expected Response'], inplace=True)
    
    return dataset


def load_experiment_from_config(config_file: Path) -> Tuple[XRIFGenerator, pd.DataFrame, Path]:
    with open(config_file, 'r') as file:
        config_dict = safe_load(file)
        test_dataset_path = config_dict['test_dataset']

        test_dataset = pd.read_csv(test_dataset_path)
        output_folder = config_dict['output_folder']

        del config_dict['output_folder']
        del config_dict['test_dataset']

        test_dataset = process_test_dataset(test_dataset)
        
        return XRIFGenerator(**config_dict), test_dataset, output_folder

def main():
    for config in os.listdir('experiment_configs'):
        xrif, test_dataset, output_folder = load_experiment_from_config(Path('experiment_configs/' + config))
        xrif.batch_test_run(test_dataset, output_folder)
        gc.collect()

if __name__ == "__main__":
    main()