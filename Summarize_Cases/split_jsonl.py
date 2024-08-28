import os

def split_jsonl_file(input_file, lines_per_file):
    # Create a directory to store the split files
    output_dir = os.path.splitext(input_file)[0] + "_split"
    os.makedirs(output_dir, exist_ok=True)

    # Open the input file
    with open(input_file, 'r') as f:
        # Initialize variables
        file_count = 1
        line_count = 0
        output_file = None

        # Iterate over each line in the input file
        for line in f:
            # Create a new output file if necessary
            if line_count % lines_per_file == 0:
                if output_file:
                    output_file.close()
                output_file = open(os.path.join(output_dir, f"split_{file_count}.jsonl"), 'w')
                file_count += 1

            # Write the line to the current output file
            output_file.write(line)
            line_count += 1

        # Close the last output file
        if output_file:
            output_file.close()

    print(f"Split {input_file} into {file_count - 1} files.")

# Usage example
split_jsonl_file('./Summarize_Cases/outcome.jsonl', 3000)