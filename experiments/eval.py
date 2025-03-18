import json 
from pathlib import Path
import ast
import os

OUTPUT_DIR = Path("test_results")
# output_folder = 'eval_test'
# eval_json_path ='test_results/all_e7_deepseek7b/Prompt_5.0_2025-03-18-04-11-35.json'
# eval_json = None

def eval_xrif(eval_json: dict, xrif_gen: dict, xrif_ex: list, response_type):
    error_message = None
    dump_json = eval_json
    xrif_ex = list(xrif_ex)
    if xrif_gen and type(xrif_gen) == dict:
        if 'actions' in xrif_gen.keys():
            actions = xrif_gen['actions']
            if len(xrif_ex) == len(actions):
                if response_type == 'Nav':
                    list_of_locations = [action['input']['name'] for action in actions if ('action' in action and action['action'] == 'navigate') and ('input' in action) and ('name' in action['input'])]
                    set_of_locations = set(list_of_locations)
                    set_of_expected_locations = set(xrif_ex)
                    if set_of_locations == set_of_expected_locations:
                        note = "All locations are present in the response"
                        dump_json['Notes'] = note
                for i in range(len(xrif_gen['actions'])):
                    if 'action' in actions[i]:
                        if actions[i]['action'] == 'navigate' :
                            if type(xrif_ex[i]) == 'str':
                                if 'input' in actions[i]:
                                    if 'name' in actions[i]['input'].keys():
                                        if actions[i]['input']['name'] == xrif_ex[i] or (type(xrif_ex[i]) == 'list' and actions[i]['input']['name'] in xrif_ex[i]):
                                            continue
                                        else:
                                            error_message = f"Expected location name: {actions[i]['name']} does not match with the provided location name: {xrif_ex[i]}"
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
                            if type(xrif_ex[i]) == 'tuple':
                                if 'input' in actions[i]:
                                        if actions[i]['action'] != xrif_ex[i][0]:
                                            error_message = f"Expected action at action object {i} : {xrif_ex[i][0]}  does not match with the provided action: {actions[i]['action']}"
                                            dump_json['Error Message'] = error_message
                                            break
                                        else:
                                            if (int(actions[i]['input']))/60 == xrif_ex[i][1]:
                                                error_message = f"Expected wait time: {actions[i]['input']} does not match with the provided wait time: {xrif_ex[i][1]}"
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
                            if type(xrif_ex[i]) == 'tuple':
                                if ('action' in actions[i]) and ('input' in actions[i]):
                                    if actions[i]['action'] == xrif_ex[i][0] and (actions[i]['input'] == xrif_ex[i][1] or (type(xrif_ex[i]) == 'list' and actions[i]['input'] in xrif_ex[i])):
                                        continue
                                    else:
                                        error_message = f"Expected speak message: {xrif_ex[i]} does not match with the generated speak message: {actions[i]['input']} or Expected action: {xrif_ex[i][0]} does not match with the generated action: {actions[i]['action']}"
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
                        error_message = f"Response Error Missing field: 'action' in action object {i}"
                        dump_json['Error Message'] = error_message
                        break
    else:
        error_message = "XRIF response is Invalid"
        dump_json['Error Message'] = error_message
    
    return dump_json



for folder in os.listdir(OUTPUT_DIR):
    if Path(OUTPUT_DIR / folder).is_dir():
        for file in os.listdir(OUTPUT_DIR / folder):
            if file.endswith('.json'):
                eval_json_path = OUTPUT_DIR / folder / file
                with open(eval_json_path, 'r') as f:
                    eval_json = json.load(f)
                if 'XRIF Generated' in eval_json and 'Expected Response' in eval_json:
                    expected_response = []
                    try:
                        expected_response = ast.literal_eval(eval_json['Expected Response'])
                    except:
                        expected_response = []
                    dump_json = eval_xrif(eval_json, eval_json['XRIF Generated'], expected_response, 'Nav')
                with open(eval_json_path, 'w') as file:
                    json.dump(dump_json, file, indent=4)