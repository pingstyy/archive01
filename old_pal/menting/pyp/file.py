fle = open("name.txt" , "X")

dt34 = fle.read()
fle = open("name.txt" , "W") # // creates /write del all content or full path to    file
fle.write("content") # it's crazy write
fle.tell() # to know where the cursor is in file

fle.seek(0) #movee the cursor to beginning of file


fle.close()

# for images
img = open("images.jpg" , "rb") # 
img_copy = open("images.jpg" , "wb") 
img_copy.write(img) 

############################################################################################
                ##################################################
                
import zipfile

def compress_to_zip(files, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in files:
            zipf.write(file)

# Example usage:
files_to_zip = ['file1.txt', 'file2.txt']
compress_to_zip(files_to_zip, 'compressed_files.zip')
#-------------------------------------------------------------------

import tarfile

def compress_to_tar(files, tar_name):
    with tarfile.open(tar_name, 'w') as tar:
        for file in files:
            tar.add(file)

# Example usage:
files_to_tar = ['file1.txt', 'file2.txt']
compress_to_tar(files_to_tar, 'compressed_files.tar')

#-------------------------------------------------------------------
import tarfile

def compress_to_tar_gz(files, tar_gz_name):
    with tarfile.open(tar_gz_name, 'w:gz') as tar_gz:
        for file in files:
            tar_gz.add(file)

# Example usage:
files_to_tar_gz = ['file1.txt', 'file2.txt']
compress_to_tar_gz(files_to_tar_gz, 'compressed_files.tar.gz')
#--------------------------------------------------------------------
import rarfile

def compress_to_rar(files, rar_name):
    with rarfile.RarFile(rar_name, 'w') as rar:
        for file in files:
            rar.write(file)

# Example usage:
files_to_rar = ['file1.txt', 'file2.txt']
compress_to_rar(files_to_rar, 'compressed_files.rar')
#--------------------------------------------------------------------
import rarfile

def compress_to_rar(files, rar_name):
    with rarfile.RarFile(rar_name, 'w') as rar:
        for file in files:
            rar.write(file)

# Example usage:
files_to_rar = ['file1.txt', 'file2.txt']
compress_to_rar(files_to_rar, 'compressed_files.rar')
#--------------------------------------------------------------------
import zipfile

def extract_zip(zip_file, extract_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# Example usage:
extract_zip('compressed_files.zip', 'extracted_files')
#------------------------------------------------------------------
import tarfile            ##             Same for GZipped files

def extract_tar(tar_file, extract_dir):
    with tarfile.open(tar_file, 'r') as tar_ref:
        tar_ref.extractall(extract_dir)

# Example usage:
extract_tar('compressed_files.tar', 'extracted_files')
#------------------------------------------------------------------
import rarfile

def extract_rar(rar_file, extract_dir):
    with rarfile.RarFile(rar_file, 'r') as rar_ref:
        rar_ref.extractall(extract_dir)

# Example usage:
extract_rar('compressed_files.rar', 'extracted_files')
#------------------------------------------------------------------
def text_to_binary(input_file, output_file):
    with open(input_file, 'r') as f:
        text_data = f.read()
    with open(output_file, 'wb') as f:
        f.write(text_data.encode('utf-8'))

# Example usage:
text_to_binary('input_text.txt', 'output_binary.bin')
#------------------------------------------------------------------
