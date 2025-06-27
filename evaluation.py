"""
Evaluasi Performa Metode Similarity untuk Information Retrieval
===============================================================

Script ini mengevaluasi performa berbagai metode similarity (Rabin-Karp dengan Dice Coefficient 
dan Cosine Similarity dengan TF-IDF) menggunakan metrik precision, recall, dan F1-score.

Metrik Evaluasi:
- Precision: TP / (TP + FP) - Proporsi dokumen relevan dari yang diambil
- Recall: TP / (TP + FN) - Proporsi dokumen relevan yang berhasil diambil
- F1-Score: 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean

Author: Evaluasi untuk Penelitian Similarity Abstraction
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

class SimilarityEvaluator:
    """
    Kelas untuk melakukan evaluasi performa metode similarity
    """
    
    def __init__(self):
        self.results = {}
        
    def parse_gold_standard(self, gold_standard_str: str) -> Set[int]:
        """
        Parse string gold standard menjadi set of document IDs
        
        Args:
            gold_standard_str: String berisi dokumen relevan (format: "41, 31" atau "44, 47, 61, 87")
            
        Returns:
            Set of document IDs
        """
        if pd.isna(gold_standard_str) or gold_standard_str == "":
            return set()
        
        # Bersihkan string dan split berdasarkan koma
        doc_ids = []
        for doc_id in str(gold_standard_str).split(','):
            doc_id = doc_id.strip()
            if doc_id and doc_id.isdigit():
                doc_ids.append(int(doc_id))
        
        return set(doc_ids)
    
    def parse_prediction_results(self, prediction_str: str) -> Set[int]:
        """
        Parse string hasil prediksi menjadi set of document IDs
        
        Args:
            prediction_str: String berisi hasil prediksi (format: "(41), (31), (52)")
            
        Returns:
            Set of document IDs
        """
        if pd.isna(prediction_str) or prediction_str == "":
            return set()
        
        # Extract numbers dalam kurung
        import re
        doc_ids = []
        matches = re.findall(r'\((\d+)\)', str(prediction_str))
        for match in matches:
            doc_ids.append(int(match))
        
        return set(doc_ids)
    
    def calculate_metrics(self, predicted: Set[int], actual: Set[int]) -> Dict[str, float]:
        """
        Hitung precision, recall, dan F1-score
        
        Args:
            predicted: Set dokumen yang diprediksi relevan
            actual: Set dokumen yang benar-benar relevan (gold standard)
            
        Returns:
            Dictionary berisi precision, recall, F1-score
        """
        if len(actual) == 0:
            # Jika tidak ada dokumen relevan di gold standard
            if len(predicted) == 0:
                return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0}
            else:
                return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        if len(predicted) == 0:
            # Jika tidak ada prediksi
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        # Hitung True Positive, False Positive, False Negative
        tp = len(predicted.intersection(actual))  # Dokumen yang diprediksi dan benar relevan
        fp = len(predicted - actual)              # Dokumen yang diprediksi relevan tapi tidak
        fn = len(actual - predicted)              # Dokumen relevan yang tidak diprediksi
        
        # Hitung metrik
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def evaluate_method(self, results_df: pd.DataFrame, method_column: str, 
                       gold_standard_column: str = 'expert') -> Dict[str, float]:
        """
        Evaluasi satu metode similarity
        
        Args:
            results_df: DataFrame berisi hasil similarity
            method_column: Nama kolom yang berisi hasil prediksi metode
            gold_standard_column: Nama kolom yang berisi gold standard
            
        Returns:
            Dictionary berisi rata-rata precision, recall, F1-score
        """
        all_metrics = []
        detailed_results = []
        
        for idx, row in results_df.iterrows():
            query_id = row['id_kueri']
            predicted = self.parse_prediction_results(row[method_column])
            actual = self.parse_gold_standard(row[gold_standard_column])
            
            metrics = self.calculate_metrics(predicted, actual)
            all_metrics.append(metrics)
            
            detailed_results.append({
                'query_id': query_id,
                'predicted_docs': predicted,
                'actual_docs': actual,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'tp': metrics.get('tp', 0),
                'fp': metrics.get('fp', 0),
                'fn': metrics.get('fn', 0)
            })
        
        # Hitung rata-rata metrik
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        avg_f1 = np.mean([m['f1_score'] for m in all_metrics])
        
        return {
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1_score': avg_f1,
            'detailed_results': detailed_results,
            'total_queries': len(results_df)
        }
    
    def evaluate_all_methods(self, rk_results_path: str = 'RK_similarity_results.xlsx',
                           cosine_results_path: str = 'Cosine_similarity_results.xlsx'):
        """
        Evaluasi semua metode similarity
        
        Args:
            rk_results_path: Path ke file hasil Rabin-Karp
            cosine_results_path: Path ke file hasil Cosine Similarity
        """
        print("="*80)
        print("EVALUASI PERFORMA METODE SIMILARITY")
        print("="*80)
        
        # Load hasil Rabin-Karp
        try:
            rk_df = pd.read_excel(rk_results_path, sheet_name='Results RK')
            print(f"✓ Berhasil load {rk_results_path}")
        except Exception as e:
            print(f"✗ Error loading {rk_results_path}: {e}")
            return
        
        # Load hasil Cosine Similarity
        try:
            cosine_df = pd.read_excel(cosine_results_path, sheet_name='Results Cosine')
            print(f"✓ Berhasil load {cosine_results_path}")
        except Exception as e:
            print(f"✗ Error loading {cosine_results_path}: {e}")
            return
        
        print()
        
        # Evaluasi setiap metode
        methods = [
            ('Rabin-Karp + Porter Stemmer', rk_df, 'rk_porter'),
            ('Rabin-Karp + Sastrawi Stemmer', rk_df, 'rk_sastrawi'),
            ('Cosine Similarity + Porter Stemmer', cosine_df, 'cosine_porter'),
            ('Cosine Similarity + Sastrawi Stemmer', cosine_df, 'cosine_sastrawi')
        ]
        
        evaluation_results = {}
        
        for method_name, df, column in methods:
            print(f"Evaluasi {method_name}...")
            results = self.evaluate_method(df, column)
            evaluation_results[method_name] = results
            
            print(f"  Precision: {results['avg_precision']:.4f}")
            print(f"  Recall:    {results['avg_recall']:.4f}")
            print(f"  F1-Score:  {results['avg_f1_score']:.4f}")
            print(f"  Total Queries: {results['total_queries']}")
            print()
        
        # Tampilkan ringkasan perbandingan
        self.display_comparison_table(evaluation_results)
        
        # Simpan hasil evaluasi detail
        self.save_detailed_results(evaluation_results)
        
        return evaluation_results
    
    def display_comparison_table(self, evaluation_results: Dict):
        """
        Tampilkan tabel perbandingan hasil evaluasi
        """
        print("="*80)
        print("TABEL PERBANDINGAN PERFORMA METODE")
        print("="*80)
        
        # Header tabel
        print(f"{'Metode':<40} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 80)
        
        # Data untuk setiap metode
        for method_name, results in evaluation_results.items():
            precision = results['avg_precision']
            recall = results['avg_recall']
            f1_score = results['avg_f1_score']
            
            print(f"{method_name:<40} {precision:<12.4f} {recall:<12.4f} {f1_score:<12.4f}")
        
        print("-" * 80)
        
        # Cari metode terbaik
        best_f1_method = max(evaluation_results.keys(), 
                           key=lambda x: evaluation_results[x]['avg_f1_score'])
        best_precision_method = max(evaluation_results.keys(),
                                  key=lambda x: evaluation_results[x]['avg_precision'])
        best_recall_method = max(evaluation_results.keys(),
                               key=lambda x: evaluation_results[x]['avg_recall'])
        
        print(f"\nMETODE TERBAIK:")
        print(f"  F1-Score terbaik:  {best_f1_method}")
        print(f"  Precision terbaik: {best_precision_method}")
        print(f"  Recall terbaik:    {best_recall_method}")
        print()
    
    def save_detailed_results(self, evaluation_results: Dict):
        """
        Simpan hasil evaluasi detail ke file Excel
        """
        output_file = 'evaluation_results.xlsx'
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Sheet ringkasan
            summary_data = []
            for method_name, results in evaluation_results.items():
                summary_data.append({
                    'Metode': method_name,
                    'Precision': results['avg_precision'],
                    'Recall': results['avg_recall'],
                    'F1-Score': results['avg_f1_score'],
                    'Total_Queries': results['total_queries']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet detail untuk setiap metode
            for method_name, results in evaluation_results.items():
                sheet_name = method_name.replace(' ', '_').replace('+', 'and')[:31]  # Excel sheet name limit
                
                detail_data = []
                for detail in results['detailed_results']:
                    detail_data.append({
                        'Query_ID': detail['query_id'],
                        'Predicted_Docs': ', '.join(map(str, sorted(detail['predicted_docs']))),
                        'Actual_Docs': ', '.join(map(str, sorted(detail['actual_docs']))),
                        'Precision': detail['precision'],
                        'Recall': detail['recall'],
                        'F1_Score': detail['f1_score'],
                        'True_Positive': detail['tp'],
                        'False_Positive': detail['fp'],
                        'False_Negative': detail['fn']
                    })
                
                detail_df = pd.DataFrame(detail_data)
                detail_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✓ Hasil evaluasi detail disimpan ke: {output_file}")

def main():
    """
    Fungsi utama untuk menjalankan evaluasi
    """
    evaluator = SimilarityEvaluator()
    
    print("Memulai evaluasi metode similarity...")
    print("Pastikan file RK_similarity_results.xlsx dan Cosine_similarity_results.xlsx tersedia")
    print()
    
    # Jalankan evaluasi
    results = evaluator.evaluate_all_methods()
    
    if results:
        print("="*80)
        print("EVALUASI SELESAI!")
        print("="*80)
        print("File yang dihasilkan:")
        print("  - evaluation_results.xlsx: Hasil evaluasi detail semua metode")
        print()
        print("Interpretasi Hasil:")
        print("  - Precision tinggi: Metode menghasilkan sedikit false positive")
        print("  - Recall tinggi: Metode berhasil menemukan sebagian besar dokumen relevan")
        print("  - F1-Score tinggi: Keseimbangan baik antara precision dan recall")

if __name__ == "__main__":
    main()
