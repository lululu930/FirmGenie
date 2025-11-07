import ast
import logging
import os
import random
import re
import time
import ast
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from typing import Dict, Any
from tqdm import tqdm  # Used for progress bars
import requests
import json
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
from data_analysis_module.Prompt_templete import *
from data_analysis_module.tools_for_data_processing import *
from data_analysis_module.wrong_sample_record import *
from data_analysis_module.accuract_calculate import *
from tools_global.tool_global import *
# from Knowledge_graph_construct.extract_info_from_database import NestedModelMatcher
from Knowledge_graph_construct.result_verify import *
# Configure logging
logging.basicConfig(
    filename='inference_process_0225.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def get_response(prompt, model_id, temperature=0, max_tokens=4096, max_retries=3, top_k=20, top_p=0.95):
    retries = 0
    while retries < max_retries:
        try:
            # Record input prompt
            # logging.info(f"Input Prompt: {prompt}")
            # response = client_SF.chat.completions.create(
            response = client_server_llama_factory.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system",
                     "content": "You are an outstanding researcher in the field of cybersecurity, especially in device identification for cyberspace mapping."},
                    {"role": "user",
                     "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                # logprobs=True,  # Enable logprobs
                # top_logprobs=5  # Return top 5 candidates for each token
                # top_p=0.85,  # Added: top-p sampling (nucleus sampling)
                # top_k=40,  # Added: top-k sampling
                # frequency_penalty=0.8,  # Reduce the frequency of frequently occurring words
                # presence_penalty=0.5,   # Reduce the frequency of words that have already appeared
                # repetition_penalty=1.2, # Repetition penalty factor
                # stop=["\n", "</s>", "###"], # Stop sequences
                # do_sample=True,          # Enable sampling
            )
            # print(response.choices[0].message.content)
            return response.choices[0].message.content, ""
        except Exception as e:
            if '429' in str(e):
                retries += 1
                wait_time = random.uniform(1, 2 ** retries)  # Exponential backoff
                print(f"Request frequency is too high, retrying after {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                retries += 1
                wait_time = random.uniform(1, 2 ** retries)  # Exponential backoff
                print(f"An error occurred: {e}")
                time.sleep(wait_time)
                # break
    else:
        print("Maximum retry attempts reached, still unable to get a response.")
        return "", "Request failed"

def load_training_data(file_path: str) -> dict:
    """Load training data file

    Args:
        file_path: JSON file path

    Returns:
        dict: Dictionary containing all data

    Raises:
        FileNotFoundError: Raised when the file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data file not found: {file_path}")

    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # for line in tqdm(f, desc="Loading data"):
        #     try:
        #         data_list.append(json.loads(line))q
        #     except json.JSONDecodeError as e:
        #         print(f"JSON parsing error: {e}, skipping this line")
        data_dict = json.load(f)

    return data_dict


# Configure retry strategy (exponential backoff)
# @retry(wait=wait_exponential(multiplier=1, min=4, max=60),
#        stop=stop_after_attempt(5),
#        retry_error_callback=lambda _: None)  # Return None on failure


def split_sample_to_chunks(banner, window_size=2500, overlap_size=100, min_chunk=50):
    """
    Split the sample into multiple chunks
    :param banner:
    :return:
    """
    WINDOW_SIZE = window_size  # Window length
    OVERLAP_SIZE = overlap_size  # Overlap size
    MIN_CHUNK = min_chunk  # Minimum processing chunk

    # Generate sliding window chunks
    chunks = []
    if len(banner) <= WINDOW_SIZE:
        # Process short text directly
        chunks = [banner]
    else:
        # chunks.append(banner[:WINDOW_SIZE])  # Add the first segment first
        # start = WINDOW_SIZE - OVERLAP_SIZE
        start = 0
        # end = 0
        while start < len(banner):
            end = min(start + WINDOW_SIZE, len(banner))
            chunk = banner[start:end]

            # Smart boundary adjustment
            if end < len(banner):
                split_chunk = banner[end - OVERLAP_SIZE:end]  # Overlap region
                last_newline = split_chunk.rfind('\n')
                if last_newline < 0:
                    last_newline = split_chunk.rfind('>')
                    if last_newline < 0:
                        last_newline = split_chunk.rfind(' ')

                # Find the last newline in the overlap region as the end of the previous chunk
                first_newline = split_chunk.find('\n')  # Find the first newline in the overlap region as the start of the next chunk
                if first_newline < 0:
                    first_newline = split_chunk.find('<')
                    if first_newline < 0:
                        first_newline = split_chunk.find(' ')
                start_new = end - OVERLAP_SIZE + first_newline
                end = end - OVERLAP_SIZE + last_newline + 1
                chunk = banner[start:end]
                start = start_new
                # last_newline = chunk.rfind('\n')
                # if last_newline > WINDOW_SIZE - 200:  # Find newline within the last 200 characters
                #     end = start + last_newline + 1
                #     chunk = banner[start:end]
            else:
                start = end
            chunks.append(chunk)
            # start = end - OVERLAP_SIZE  # Set overlap region
    return chunks


def collect_response(banner, prompt, model_id):
    """Smart chunking for long text, maintaining context continuity"""
    # Initialize result container
    combined_json = {"brand": [], "model": [], "version": []}
    inferences = []
    chunks = split_sample_to_chunks(banner, window_size=2000)
    # Process each chunk
    scored_plus_items = [] # Record items extracted together, add base score
    candidate_brand = []
    candidate_model = []
    candidate_version = []
    keyword_context_list = []
    # valid_result = []  # Save the correct brand/model/version combinations after verification
    # print(f"！！！！！*********************Text chunking, {len(chunks)} chunks in total*********************！！！！！")
    print(len(chunks))
    for i, chunk in enumerate(chunks):
        # if len(chunk) < MIN_CHUNK and i != 0:
        #     continue  # Skip overly small tail segments
        # Remove tabs
        # if i < 1:
        #     continue
        chunk = remove_ntr(chunk)
        # chunk = chunk[50:]
        # Add chunk position marker
        tagged_chunk = f"[Part {i + 1}/{len(chunks)}]\n{chunk}"
        # Get response
        #########Round 1: Get LLM response
        # print(f"\n*********************Chunk {i}, starting to get LLM response*********************")
        prompt_model = PROMPT_with_scoring_llm_with_criterion + '\n' + 'banner:' + tagged_chunk + '\n\n' + 'candidate_brand:' + str(candidate_brand) + '\n' + 'candidate_model:' + str(candidate_model) + '\n' + 'candidate_version:' + str(candidate_version)
        result, inference = get_response(prompt_model, model_id)
        if result:
            if inference != "":
                inferences.append(inference)
            else:
                inferences.append(result)
            try:
                current_json = extract_json(result)
                # print("init_result")
            except Exception as e:
                print(f"Error: {e}")
                current_json = {}
                continue
        else:
            print("Error: No response from the API.")
            continue
        # Unify key-value format
        try:
            if "firmware_version" in current_json.keys():
                tmp = {"brand": current_json["brand"], "model": current_json["model"], "version": current_json["firmware_version"]}
                current_json = tmp
            else:
                if "version" in current_json.keys():
                    tmp = {"brand": current_json["brand"], "model": current_json["model"],
                           "version": current_json["version"]}
                    current_json = tmp
                else:
                    current_json["version"] = []
                    current_json["firmware_version"] = []
                    tmp = {"brand": current_json["brand"], "model": current_json["model"],
                           "version": current_json["version"]}
                    current_json = tmp
        except KeyError:
            if 'brand' not in current_json.keys():
                current_json["brand"] = []
            if 'model' not in current_json.keys():
                current_json["model"] = []
            if "firmware_version" not in current_json.keys():
                current_json["firmware_version"] = []
            tmp = {"brand": current_json["brand"], "model": current_json["model"],
                   "version": current_json["firmware_version"]}
            current_json = tmp
        # print(f"Chunk {i}, Round 1 LLM result: {current_json}")
        # Update candidate keywords and extract context
        for brand in current_json["brand"]:
            try:
                if brand[0].lower() not in candidate_brand and isinstance(brand[0], str) and len(brand[0]) > 1:
                    candidate_brand.append(str(brand[0]).lower())
            except:
                pass
        for model in current_json["model"]:
            try:
                if model[0].lower() not in candidate_model and isinstance(model[0], str) and len(model[0]) > 3:
                    candidate_model.append(str(model[0]).lower())
            except:
                pass
        for version in current_json["version"]:
            try:
                if version[0].lower() not in candidate_version and isinstance(version[0], str) and len(version[0]) > 3:
                    candidate_version.append(str(version[0]).lower())
            except:
                pass
        # print(f"Candidate keywords updated: {candidate_brand}, {candidate_model}, {candidate_version}")
        # Extract context
        # print("\n*********************Extracting keyword context*********************")
        optimized_segments = extract_segments_with_smart_boundaries(
            candidate_brand + candidate_model + candidate_version,
            chunk,
            context_length=80,
            merge_distance=30,
            use_sentence_boundaries=True
        )
        keyword_context_list += optimized_segments
        for cat in ['brand', 'model', 'version']:

            if val := current_json.get(cat):
                # Confidence calculation based on position and repetition count
                for val_item in val:
                    if val_item == [] or val_item == "" or val_item == ['']:
                        continue
                    if isinstance(val_item, int) or isinstance(val_item, float):
                        continue
                    # if cat == "brand" and val_item in candidate_brand:
                    #     continue
                    # if cat == "model" and val_item in candidate_model:
                    #     continue
                    # if cat == "version" and val_item in candidate_version:
                    #     continue
                    # val_item = str(val_item[0]).lower()
                    # weight = 1 / (i + 1) + combined_json[cat].count(val_item) * 0.2   # .count(val) iterates the list to count occurrences where the first element of the tuple equals val
                    if len(val_item) == 2 and (isinstance(val_item[1], float) or isinstance(val_item[1], int)):
                        weight = val_item[1]  # Directly use the confidence from ds output as weight
                    else:
                        try:
                            if len(val) == 2 and isinstance(val[0][0], str) and (isinstance(val[1][0], float) or isinstance(val[1][0], int)):  # If val is a tuple and the second element is a float, use the second element as weight
                                weight = val[1][0]
                                if (str(val[0][0]).lower(), weight) not in combined_json[cat]:
                                    combined_json[cat].append((str(val[0][0]).lower(), weight))
                                break
                            else:
                                weight = 0
                        except:
                            if val[0] == [] and val[1] == []:
                                if isinstance(val[1][0], float) or isinstance(val[1][0], int):
                                    weight = val[1][0]
                                else:
                                    weight = 0
                                if ("", weight) not in combined_json[cat]:
                                    if ("", weight) not in combined_json[cat]:
                                        combined_json[cat].append(("", weight))
                                break
                            elif isinstance(val[1], float) or isinstance(val[1], int):
                                weight = val[1]
                    if (str(val_item[0]).lower(), weight) not in combined_json[cat]:
                        combined_json[cat].append((str(val_item[0]).lower(), weight))
        # print(f"Aggregated result: {combined_json}")
        time.sleep(1)  # API rate control


    # print("\n!!!!!*********************Result merging analysis*********************！！！！！")
    # Consolidate parts with keywords
    segment_banner = ""
    for item in keyword_context_list:
        text = item["text_segment"]
        segment_banner = segment_banner + text + '\n'
    # Second verification based on knowledge base information
    # Create matcher
    # todo: Based on output confidence, decide if KB verification is needed? Improve efficiency
    # print("*********************Second verification based on knowledge base information*********************")
    validator = TripletValidator(database_brand, database_model)
    triplet_results, model_match_results, version_match_results = validator.validate_triplets(
        candidate_brand,
        candidate_model,
        candidate_version,
        confidence_threshold=0.7
    )
    # print(f"Found {len(triplet_results)} valid combinations:\n{triplet_results}")

    # Sort by confidence
    triplet_results = sort_by_confidence_desc(triplet_results)
    # No combination found
    # Combination found but with null values (to prevent version errors)
    verify_flag = False

    if len(triplet_results) >= 1:
        possible_combination = triplet_results[0]
        # Check for null values
        if possible_combination.brand == "" or possible_combination.model == "" or possible_combination.version == "":
            # print("Combination found but with null values")
            verify_flag = True
            if all(match.version_list == '[]' for matches in model_match_results.values() for match in matches if hasattr(match, 'version_list')):
                verify_flag = False
    KB_verified_results = {}
    KB_verified_inference = ""
    KB_verified_results_init = ""
    if len(triplet_results) < 1 or verify_flag:
        ## LLM re-verification
        # print("*********************LLM re-verification - based on knowledge base*********************")
        candidate_verify = []
        for brand_kb, model_list_kb in model_match_results.items():
            if model_list_kb == []:
                pass
            else:
                for model_kb in model_list_kb:
                    hint = {"candidate_model": model_kb.original, "similar_model": model_kb.matched, "similarity": model_kb.confidence, "brand": model_kb.brand, "similar_model_version_list": json.loads(model_kb.version_list)}
                    if hint not in candidate_verify:
                        candidate_verify.append(hint)
        # for item_brand in candidate_brand:
        #     for item_model in candidate_model:
        #
        #         if item_brand in database_brand.keys() and item_model in database_brand[item_brand].keys():
        #             collected_versions = database_brand[item_brand][item_model]
        #             if item_model.startswith(item_brand):
        #                 item_model = item_model[len(item_brand):].strip().lower()
        #             if item_model in database_brand[item_brand].keys():
        #                 collected_versions += database_brand[item_brand][item_model]
        #
        #             if collected_versions:
        #                 tmp_item = {"brand": item_brand, "model": item_model, "collected_versions_in_KB": collected_versions}
        #                 if tmp_item not in candidate_verify:
        #                     candidate_verify.append(tmp_item)
        if candidate_verify:
            if len(chunks) > 1:
                prompt_verify_kb = PROMPT_workfolw_verify + '\n' + 'concat_banner:' + segment_banner + '\n\n' + '\n\n' + 'Knowledge_Base_Info' + json.dumps(candidate_verify)
            else:
                prompt_verify_kb = PROMPT_workfolw_verify + '\n' + 'concat_banner:' + remove_ntr(chunks[0]) + '\n\n' + '\n\n' + 'Knowledge_Base_Info' + json.dumps(candidate_verify)
            KB_verified_results_init, KB_verified_inference = get_response(prompt_verify_kb, model_id)  # + 'candidate_brand: ' + str(candidate_brand) + '\n' + 'candidate_model:' + str(candidate_model) + '\n' + 'candidate_version:' + str(candidate_version)
            try:
                KB_verified_results = extract_json(KB_verified_results_init)

            except Exception as e:
                print(f"Error: {e}")
                KB_verified_results = {}
        # print("\nLLM re-verification - based on knowledge base result:", KB_verified_results)
    else:
        if len(triplet_results) > 0:
            KB_verified_results = {"brand": [triplet_results[0].brand, triplet_results[0].confidence], "model": [triplet_results[0].model, triplet_results[0].confidence], "version": [triplet_results[0].version, triplet_results[0].confidence]}


    # Second verification
    # todo: Change order? Move before knowledge base verification
    # print("*********************LLM re-verification - without knowledge base*********************")
    verified_json = {}
    verified_inference = ""
    # if segment_banner:
    #     prompt_verify = PROMPT_workflow_double_check_with_scoring + '\n' + 'concat_banner:' + segment_banner + '\n\n' + 'candidate_brand: ' + str(candidate_brand) + '\n' + 'candidate_model:' + str(candidate_model) + '\n' + 'candidate_version:' + str(candidate_version)
    #     verified_inference, inference = get_response(prompt_verify, model_id)
    #
    # if verified_inference:
    #     try:
    #         verified_json = extract_json(verified_inference)
    #         print("verified_json:", verified_json)
    #     except Exception as e:
    #         print(f"Error: {e}")
    # print(f"LLM re-verification - without knowledge base result: {verified_json}")

    # print("*********************LLM round 1 output aggregation - outputting highest score*********************")
    final_output = {}
    for cat, candidates in combined_json.items():
        # Weighted voting algorithm
        score_board = defaultdict(float)
        for value, weight in candidates:
            # for value_item in value:
            # score_board[value] += weight
            if value not in score_board.keys():
                score_board[value] = weight
            else:
                if weight > score_board[value]:
                    score_board[value] = weight

        if score_board:
            final_output[cat] = [max(score_board.items(), key=lambda x: x[1])[0], max(score_board.items(), key=lambda x: x[1])[1]]
        else:
            final_output[cat] = ["", 0]
    try:
        final_inference = '@@@@@'.join(inferences)
    except:
        final_inference = ""
    # print(f"LLM round 1 output result: {final_output}")
    rescored_combined_result = {}
    # print("*********************Rescoring with scoring function*********************")
    final_output_rescore, rescored_combined_result = score_extracted_keywords(remove_ntr(banner), combined_json, scored_plus_items)
    # print(f"Rescoring with scoring function result: {final_output_rescore}")
    return final_inference, final_output, final_output_rescore, rescored_combined_result, combined_json, verified_json, verified_inference, triplet_results, model_match_results, KB_verified_results, KB_verified_results_init


    # Process response result
    # output = result.split('\n')[-1].split('<')[-1].split('>')[0].strip()
    # match = re.search(r'json\s*([^`]*)\s*', result, re.DOTALL)


def process_data(data_dict: dict, result_path, model_id, prompt):
    """Process data and generate responses

    Args:
        data_dict: Dictionary of training data
    """
    brand_flag = False
    for brand, data_list in data_dict.items():
        print(brand)
        brand_data_list = []
        for data in tqdm(data_list, desc="Processing data"):
            try:
                # Identification
                index = data['index']
                if index in wrong_version_record_1 or index in duplicate_sample:
                    continue
                if index in wrong_version_record_4 or index in no_version_wrong_sample_record:
                    continue

                # Filter invalid banners
                banner_init = data['banner'].strip()
                # if not is_banner_useful(banner_init):
                #     print("status code error")
                #     continue
                # Correct labels
                model = data["label"][1]
                version = data["label"][2]
                data["label"][0], data["label"][1], data["label"][2] = hard_label_in_instruction_data(index, brand,
                                                                                                      model,
                                                                                                      version)
                data["label"][1], data["label"][2] = correct_wrong_sample(index, data["label"][1], data["label"][2])
                data["label"][0], data["label"][1], data["label"][2] = correct_wrong_sample_2(index, data["label"][0],
                                                                                              data["label"][1],
                                                                                              data["label"][2])
                # Preprocess banner
                banner = sanitize_network_info(banner_init, brand)
                if len(banner) > 30000 or len(banner) < 25:
                    print(f"banner too long or too short: {len(banner)}, index: {index}")
                    continue

                # Get response
                print(index)
                inference, output, rescore_output, combined_json, init_json, verified_json, verified_inference, \
                    knowledge_verify_triplets, knowledge_verify_results, KB_verified_llm_results, KB_verified_llm_inference = collect_response(banner, prompt, model_id)
                ## There are 7 types of tests in total--including original, applying scoring function, with/without confidence, with/without context, LLM re-verification, knowledge base verification, and knowledge base + LLM verification
                # 1. inference, output, init_json represent the model's normal reasoning process, output result, and the aggregated result after chunking (with/without confidence)
                # 2. rescore_output, combined_json represent the output and aggregated result after rescoring with the scoring program
                # 3. verified_json, verified_inference represent the result after LLM re-verification, input is the concatenated segments and candidate keywords
                # 4. knowledge_verify_triplets, knowledge_verify_results represent the result after knowledge base verification, purely code-based verification
                # 5. KB_verified_llm_results, KB_verified_llm_inference represent the result after knowledge base verification, LLM verification, input is concatenated segments, candidate keywords, and information extracted from the knowledge base

                if not inference:
                    print(f"Error! no output, check data: {inference}, index:{index}")
                    continue

                def convert(obj):
                    if is_dataclass(obj):
                        return asdict(obj)  # dataclass to dict
                    elif isinstance(obj, dict):
                        return {k: convert(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert(i) for i in obj]
                    else:
                        return obj
                data["inference"] = inference
                data["no_scoring_result"] = init_json
                data["code_rescored_result"] = combined_json
                data["llm_verified_result"] = verified_json
                data["llm_verified_inference"] = verified_inference
                data["knowledge_verify_triplets"] = json.dumps(
                    [asdict(item) for item in knowledge_verify_triplets],
                    ensure_ascii=False, indent=2
                )
                data["knowledge_verify_results"] = json.dumps(
                    convert(knowledge_verify_results),
                    ensure_ascii=False, indent=2
                )
                data["KB_verified_llm_results"] = KB_verified_llm_results
                data["KB_verified_llm_inference"] = KB_verified_llm_inference
                if "brand" in str(output):
                    data["DS_result"] = output
                else:
                    data["DS_result"] = []
                if "brand" in str(rescore_output):
                    data["code_rescored_result_output"] = rescore_output
                print(output)


                brand_data_list.append(data)
                # logging.info(f"Inference:\n{result}\n\n\n\n")
                time.sleep(1)

                with open(result_path, 'a+', encoding='utf-8') as json_file:
                    json_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            #         # logging.info(f"Inference_clarify:\n{result}\n\n\n\n")
            #         time.sleep(2)

            except Exception as e:
                if "rate limiting" in str(e):
                    print("Rate limiting triggered, retrying automatically...")
                    raise  # Trigger tenacity retry
                else:
                    print(f"Error occurred while processing data: {e}")
                    if 'unknown extension' in e:
                        print('check unknown extension !')
                    continue


def main(model, model_folder,  data_file, version_f=True, Prompt=PROMPT_OPTIMIZED_0404):
    print(model)
    data_f = data_file.split('sample_restore/')[-1].split('.')[0]
    father_dir = data_file.split("sample_restore")[0]
    out_f = data_f + '_result_' + model + '.json'
    output_result_path = os.path.join(father_dir, "result_record", model_folder, out_f)
    accuracy_record_path = os.path.join(father_dir, 'accuracy_record.txt')
    try:
        training_data = load_training_data(data_file)
        process_data(training_data, output_result_path, model_id_llama_factory, prompt=Prompt)
    except Exception as e:
        print(f"Program failed to run: {e}")

    model_error_path = os.path.join(father_dir, "result_record", model_folder, data_f) + '_' + model + '_model_error.json'#f'./data_analysis_module/data/hasVersion/hasVersion_filtered_version_deduplicated_result_{model}_model_error.json'
    version_error_path = os.path.join(father_dir, "result_record", model_folder, data_f) + '_' + model + '_version_error.json'#f'./data_analysis_module/data/hasVersion/hasVersion_filtered_version_deduplicated_result_{model}_version_error.json'
    brand_error_path = os.path.join(father_dir, "result_record", model_folder, data_f) + '_' + model + '_brand_error.json'
    if version_f:
        calculate_accuracy(accuracy_record_path, output_result_path, data_file, model_error_path, version_error_path, model, brand_error_sample=brand_error_path)
    else:
        calculate_accuracy_negative(accuracy_record_path, output_result_path, data_file, model_error_path, version_error_path, model, brand_error_sample=brand_error_path)


if __name__ == "__main__":

    client_server_llama_factory = OpenAI(
        base_url="your server url",
        api_key="llama_factory"
    )

    model_id_llama_factory = 'IFT_Qwen2___5_7B_Instruct_compressed_cot_0825_epoch5_6144'# 'Meta-Llama-3-8B-Instruct'  # DeepSeek-R1-Distill-Qwen-14B

    model_folder = "your model path"
    prompt_index = "whatever you want to label the output"
    data_index = "whatever you want to label the output"
    model = "_".join([model_folder, prompt_index, data_index])# Qwen2_7B_Instruct  Qwen2___5_7B_Instruct  Qwen3_Embedding_8B IFT_Llama3_8B_Instruct_0603_epoch6_0404_ch_0509 DeepSeek_R1_Distill_Qwen_14B"Instruct_finetuned_Llama3_8B_Instruct"#"deepseek_distill_llama_8b_0327"#"deepseek8b_5"  # "qwen2_7b" "deepseek_full"  RLFT_Qwen2___5_3B_Instruct_ppo_4096_0710_0404_ch_0509
    all_data_path = ''
    # test_data_path_wild = './data_analysis_module/data/hasVersion/sample_restore/hasVersion_filtered_version_0509_deduplicated.json'
    prompt = PROMPT_TEST_0826
    with open('knowledge_base_path_brand', 'r', encoding='utf-8') as f:
        database_brand = json.load(f)
    with open('knowledge_base_path_model', 'r', encoding='utf-8') as f:
        database_model = json.load(f)

    main(model, model_folder, all_data_path, version_f=True, Prompt=prompt)