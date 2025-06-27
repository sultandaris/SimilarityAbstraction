# Demo Evaluasi Metode Similarity
# =================================

import pandas as pd
import numpy as np
import re

# Contoh data dummy untuk demonstrasi
dummy_rk_results = {
    'id_kueri': [1, 2, 3],
    'rk_porter': ['(41), (31)', '(44), (47)', '(10), (20), (30)'],
    'rk_sastrawi': ['(41), (31)', '(44), (47), (61)', '(10), (20)'],
    'expert': ['41, 31', '44, 47, 61, 87', '10, 20']
}

dummy_cosine_results = {
    'id_kueri': [1, 2, 3],
    'cosine_porter': ['(41), (31), (50)', '(44), (47)', '(10)'],
    'cosine_sastrawi': ['(41)', '(44), (47), (61)', '(10), (20), (25)'],
    'expert': ['41, 31', '44, 47, 61, 87', '10, 20']
}

# Fungsi evaluasi (copy dari file utama)
def parse_gold_standard(gold_standard_str):
    if pd.isna(gold_standard_str) or gold_standard_str == "":
        return set()
    
    doc_ids = []
    for doc_id in str(gold_standard_str).split(','):
        doc_id = doc_id.strip()
        if doc_id and doc_id.isdigit():
            doc_ids.append(int(doc_id))
    
    return set(doc_ids)

def parse_prediction_results(prediction_str):
    if pd.isna(prediction_str) or prediction_str == "":
        return set()
    
    doc_ids = []
    matches = re.findall(r'\((\d+)\)', str(prediction_str))
    for match in matches:
        doc_ids.append(int(match))
    
    return set(doc_ids)

def calculate_metrics(predicted, actual, total_docs=100):
    if len(actual) == 0:
        if len(predicted) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'accuracy': 1.0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': total_docs}
        else:
            fp = len(predicted)
            tn = total_docs - fp
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'accuracy': tn/(tn+fp), 'tp': 0, 'fp': fp, 'fn': 0, 'tn': tn}
    
    if len(predicted) == 0:
        fn = len(actual)
        tn = total_docs - fn
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'accuracy': tn/(tn+fn), 'tp': 0, 'fp': 0, 'fn': fn, 'tn': tn}
    
    tp = len(predicted.intersection(actual))
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    tn = total_docs - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

# Demo evaluasi
print("="*60)
print("DEMO EVALUASI METODE SIMILARITY")
print("="*60)

# Buat DataFrame dummy
rk_df = pd.DataFrame(dummy_rk_results)
cosine_df = pd.DataFrame(dummy_cosine_results)

print("\nData RK Results:")
print(rk_df)
print("\nData Cosine Results:")
print(cosine_df)

# Demo evaluasi per query
print("\n" + "="*60)
print("EVALUASI DETAIL PER QUERY")
print("="*60)

for idx, row in rk_df.iterrows():
    query_id = row['id_kueri']
    print(f"\n--- Query {query_id} ---")
    
    # Parse gold standard
    actual = parse_gold_standard(row['expert'])
    print(f"Gold Standard: {sorted(actual)}")
    
    # Evaluasi RK Porter
    predicted_rk_porter = parse_prediction_results(row['rk_porter'])
    metrics_rk_porter = calculate_metrics(predicted_rk_porter, actual, total_docs=100)
    print(f"RK Porter: {sorted(predicted_rk_porter)}")
    print(f"  P: {metrics_rk_porter['precision']:.3f}, R: {metrics_rk_porter['recall']:.3f}, F1: {metrics_rk_porter['f1_score']:.3f}, Acc: {metrics_rk_porter['accuracy']:.3f}")
    print(f"  TP: {metrics_rk_porter['tp']}, FP: {metrics_rk_porter['fp']}, FN: {metrics_rk_porter['fn']}, TN: {metrics_rk_porter['tn']}")
    
    # Evaluasi RK Sastrawi
    predicted_rk_sastrawi = parse_prediction_results(row['rk_sastrawi'])
    metrics_rk_sastrawi = calculate_metrics(predicted_rk_sastrawi, actual, total_docs=100)
    print(f"RK Sastrawi: {sorted(predicted_rk_sastrawi)}")
    print(f"  P: {metrics_rk_sastrawi['precision']:.3f}, R: {metrics_rk_sastrawi['recall']:.3f}, F1: {metrics_rk_sastrawi['f1_score']:.3f}, Acc: {metrics_rk_sastrawi['accuracy']:.3f}")
    print(f"  TP: {metrics_rk_sastrawi['tp']}, FP: {metrics_rk_sastrawi['fp']}, FN: {metrics_rk_sastrawi['fn']}, TN: {metrics_rk_sastrawi['tn']}")

print("\n" + "="*60)
print("INTERPRETASI CONTOH")
print("="*60)

print("\nQuery 1 (Gold Standard: [31, 41]):")
print("- RK Porter [31, 41]: Perfect match → P=1.0, R=1.0, F1=1.0")
print("- RK Sastrawi [31, 41]: Perfect match → P=1.0, R=1.0, F1=1.0")

print("\nQuery 2 (Gold Standard: [44, 47, 61, 87]):")
print("- RK Porter [44, 47]: Hanya 2 dari 4 → P=1.0, R=0.5, F1=0.667")
print("- RK Sastrawi [44, 47, 61]: 3 dari 4 → P=1.0, R=0.75, F1=0.857")

print("\nQuery 3 (Gold Standard: [10, 20]):")
print("- RK Porter [10, 20, 30]: 1 false positive → P=0.667, R=1.0, F1=0.8")
print("- RK Sastrawi [10, 20]: Perfect match → P=1.0, R=1.0, F1=1.0")

print("\n" + "="*60)
print("KESIMPULAN DEMO")
print("="*60)
print("1. Metrik precision mengukur akurasi prediksi")
print("2. Metrik recall mengukur kelengkapan hasil")
print("3. F1-score memberikan nilai keseimbangan")
print("4. Accuracy mengukur proporsi prediksi yang benar")
print("5. Analisis TP, FP, FN, TN membantu memahami kinerja detail")
print("6. Evaluasi ini dapat diterapkan pada dataset sesungguhnya")

print(f"\n{'='*60}")
print("Untuk dataset sesungguhnya, jalankan notebook process.ipynb")
print("atau evaluation.ipynb setelah file hasil similarity tersedia.")
print(f"{'='*60}")
