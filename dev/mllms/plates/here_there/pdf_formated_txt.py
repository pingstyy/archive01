import re

# Open the input text file
with open('output_RAW.txt', 'r', encoding='utf-8') as text_file:
    text = text_file.read()

# Define patterns to match unwanted content
preface_pattern = r'(?i)preface|introduction|foreword'
appendix_pattern = r'(?i)appendix|index|glossary|bibliography'
random_pattern = r'(?i)random'

# Replace unwanted content with an empty string
cleaned_text = re.sub(preface_pattern, '', text, flags=re.MULTILINE)
cleaned_text = re.sub(appendix_pattern, '', cleaned_text, flags=re.MULTILINE)
cleaned_text = re.sub(random_pattern, '', cleaned_text, flags=re.MULTILINE)

# Write the cleaned text to a new file
with open('cleaned_output.txt', 'w', encoding='utf-8') as cleaned_file:
    cleaned_file.write(cleaned_text)