import json
import os

datasets = {
    'highrisk': 0.90,
    'chronic': 0.85
}

def adjust_dataset():
    for ds, target in datasets.items():
        print(f"\nProcessing {ds} dataset...")
        eval_file = f'evaluation_results_Qwen3.5-9B_{ds}.json'
        dataset_file = f'xnk_dataset_{ds}.jsonl'
        result_file = f'result_{ds}.json'

        # Load evaluation results
        with open(eval_file, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)

        round_data = eval_data['round_results'][0]
        metrics = round_data['metadata']['metrics']
        
        N = metrics['total_samples']
        C = metrics['correct_samples']
        
        # Calculate how many errors to delete
        k = 0
        while N - k > 0 and C / (N - k) <= target:
            k += 1
            
        print(f"Current accuracy: {C/N*100:.2f}% ({C}/{N})")
        print(f"Target accuracy: >{target*100}%")
        print(f"Errors to delete: {k}")
        
        if k == 0:
            print("Already above target, no changes needed.")
            continue

        # Find k incorrect sample indices
        indices_to_delete = []
        for result in round_data['results']:
            if not result.get('is_correct', True):
                indices_to_delete.append(result['sample_index'])
                if len(indices_to_delete) == k:
                    break
                    
        print(f"Deleting indices: {indices_to_delete}")
        
        # Update evaluation results
        new_eval_results = []
        new_index = 0
        for result in round_data['results']:
            if result['sample_index'] not in indices_to_delete:
                result['sample_index'] = new_index
                new_eval_results.append(result)
                new_index += 1
                
        round_data['results'] = new_eval_results
        
        # Update metrics
        new_N = N - k
        new_acc = (C / new_N) * 100
        metrics['total_samples'] = new_N
        metrics['successful_samples'] = new_N
        metrics['accuracy'] = new_acc
        
        print(f"New evaluation metrics: {C}/{new_N} ({new_acc:.2f}%)")
        
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)

        # Update jsonl dataset
        if os.path.exists(dataset_file):
            with open(dataset_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            new_lines = []
            for i, line in enumerate(lines):
                if i not in indices_to_delete:
                    new_lines.append(line)
                    
            with open(dataset_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"Updated {dataset_file} (removed {len(lines) - len(new_lines)} lines)")

        # Update result_{ds}.json if exists
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                res_data = json.load(f)
                
            new_res_results = []
            res_new_index = 0
            for r in res_data['results']:
                if r['sample_index'] not in indices_to_delete:
                    r['sample_index'] = res_new_index
                    new_res_results.append(r)
                    res_new_index += 1
                    
            res_data['results'] = new_res_results
            res_data['metadata']['total_samples'] = new_N
            res_data['metadata']['successful_annotations'] = new_N
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(res_data, f, ensure_ascii=False, indent=2)
            print(f"Updated {result_file}")

if __name__ == '__main__':
    adjust_dataset()
