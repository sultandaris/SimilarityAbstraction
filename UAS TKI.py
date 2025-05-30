import os
import re
from collections import Counter
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

print("Kelompok UAS TKI")

folder_path = r"C:\Users\Sultan Daris\Downloads\UAS TKI Projek\DokumenAbstrak\Dokumen" 

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            words = content.split()
            first_10_words = ' '.join(words[:3])
            print(f"file {filename}:\n{first_10_words}\n")
            processed_documents = {}

stemmer = StemmerFactory().create_stemmer()

for filename in os.listdir(folder_path):
    print(f"Preprocessing file: {filename}")
    file_path = os.path.join(folder_path, filename)

    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
                    
            content = content.lower()
            
            # Segmentasi teks - pisahkan menjadi kalimat berdasarkan tanda titik
            sentences = re.split(r'[.!?]+', content)
            
            content = re.sub(r'[^\w\s]', ' ', content)
            
            content = re.sub(r'\s+', ' ', content).strip()
            
            words = content.split()
            
            stop_words = {'dan', 'atau', 'yang', 'adalah', 'ini', 'itu', 'dengan', 'untuk', 'pada', 'dalam', 'dari', 'ke', 'di', 'akan', 'dapat', 'juga', 'tidak', 'ada', 'satu', 'dua', 'tiga'}
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            stemmed_words = [stemmer.stem(word) for word in filtered_words]
            
            processed_documents[filename] = stemmed_words
            
            segmented_lines = []
            for i in range(0, len(stemmed_words), 7):
                line = ' '.join(stemmed_words[i:i+7])
                segmented_lines.append(line)
            
            segmented_output = '\n'.join(segmented_lines)
            
            output_filename = f"preprocessed_{filename}"
            output_path = os.path.join(r"C:\Users\Sultan Daris\Downloads\UAS TKI Projek\DokumenAbstrak\DokumenOlahan", output_filename)
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(segmented_output)
            
            print(f"content of {filename} after preprocessing and stemming:")
            print(segmented_output)
            print(f"Original words: {len(words)}")
            print(f"After preprocessing: {len(filtered_words)}")
            print(f"After stemming: {len(stemmed_words)}")
            unique_words = len(set(stemmed_words))
            print(f"Unique words: {unique_words}")
            print(f"Saved to: {output_filename}")
            print("\n")
