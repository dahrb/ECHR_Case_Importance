import re

def fix_jsonl_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line
            #print("line: ",line)
            if not re.search(r'\"\\n\}"[^"]*"ref', line):
                #print(line)
                #print('here\n')
                line = re.sub(r'\", \"refusal\"', r'.\\"\\n}", "refusal"', line)
                #print(line)
            #Replace ", "refusal" with ."\n}", "refusal" if it isn't already ."\n}", "refusal"
            #line = re.sub(r'(?<!\.\n})", "refusal"', r'.\"\n}", "refusal"', line)
            #line = re.sub(r'(.+)", "refusal": null}', r'\1.\"\n}", "refusal": null}', line)
            #print("line 2: ",line)
            outfile.write(line)

            #application inadmissible.\"\n}", "refusal"#': null},
            #application inadmissible.\"\n}."}, "refusal": null}

input_file = '/users/sgdbareh/volatile/ECHR_Importance/Summarize_Cases/Results/split_2.jsonl'
output_file = '/users/sgdbareh/volatile/ECHR_Importance/Summarize_Cases/Results/split_2_fixed.jsonl'
fix_jsonl_file(input_file, output_file)