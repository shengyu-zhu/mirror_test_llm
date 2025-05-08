###################### 
# Main Result
###################### 
python code/core/step0_generate_story_prompt.py
python code/core/process_story_seed.py
python code/core/step1_story_generation_by_m1.py
python code/core/process_m1_story.py
python code/core/step2_add_mark.py
python code/core/process_m2_mark.py
python code/core/step3_recogonition.py
python code/core/step3_recogonition.py --process-all-m3-variants
python code/core/merge_data_m3_prompt_variants.py
python code/core/process_m3_output.py
python code/core/step3_recognition_m3.py --evaluator all

###################### 
# Analysis
###################### 
python code/analysis/accuracy_table.py
python code/analysis/randomization_check.py
python code/analysis/accuracy_by_sentence_num.py
python code/analysis/sentence_num_distribution.py
python code/analysis/step3_alternative_prompts.py
python code/analysis/m3_analysis.py

###################### 
# Additional Analysis for Robustness
###################### 
python code/robustness_check/effect_of_num_of_sentences/generate_date_set_from_story_seed.py
python code/robustness_check/effect_of_num_of_sentences/effect_of_num_of_sentences.py
python code/robustness_check/effect_of_num_of_sentences/analysis.py
python code/robustness_check/effect_of_num_of_sentences/analysis_predicted_position.py
python code/robustness_check/effect_of_num_of_sentences/analysis_distribution_of_actual_modified_sentence_position.py
python code/robustness_check/temperature_analysis/effect_of_temperature_parameters.py
python code/robustness_check/temperature_analysis/temperature_number_as_result.py
python code/robustness_check/temperature_analysis/analysis.py

###################### 
# Utilities
###################### 
# code/utils/api_keys.py
# code/utils/util.py
python code/utils/examine_data.py --examples 50