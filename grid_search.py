import itertools
import pandas as pd
from training import train_model
from config import RESULTS_DIR, OUTPUT_DIR, GIF_PATH
import imageio
import matplotlib.pyplot as plt
import os

def grid_search(train_dataset, val_images_tf, test_images_tf, train_images, val_images, test_images):
    param_grid = {
        'latent_dim': [50, 100, 150], #latent space dimensions
        'generator_filters': [64, 128], #Number of filters in generators
        'dropout_rate': [0.3, 0.5], #Dropout rate in gen and discriminator
        'learning_rate': [1e-4, 3e-4]
    }
    keys = param_grid.keys()
    combinations = list(itertools.product(*[param_grid[key] for key in keys])) #Generating all combinations
    all_results = []
    selected_combinations = combinations[:3]  # Limit to first 3 combinations
    for i, combination in enumerate(selected_combinations):
        params = dict(zip(keys, combination))
        run_name = f"run_{i+1}_ld{params['latent_dim']}_gf{params['generator_filters']}_dr{params['dropout_rate']}_lr{params['learning_rate']}"
        print(f"\n======= Starting Run {i+1}/{len(selected_combinations)} =======")
        print(f"Parameters: {params}")
        results = train_model(
            train_dataset,
            val_images_tf,
            test_images_tf,
            train_images,
            val_images,
            test_images,
            params['latent_dim'],
            params['generator_filters'],
            params['dropout_rate'],
            params['learning_rate'],
            run_name
        )
        all_results.append(results)
    results_df = pd.DataFrame(all_results) #saving as csv
    results_df.to_csv(os.path.join(RESULTS_DIR, 'grid_search_results.csv'), index=False)
    best_model = results_df.loc[results_df['test_fid'].idxmin()] #indentifying best model based on FID
    print("\n===== Grid Search Results =====")
    print(f"Best model based on test FID: {best_model['run_name']}")
    print(f"Test FID: {best_model['test_fid']}")
    print("\nAll results:")
    print(results_df)
    # Plotting comparative results
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.bar(results_df['run_name'], results_df['test_fid'])
    plt.ylabel('Test FID Score (lower is better)')
    plt.title('Test FID Score by Model Configuration')
    plt.xticks(rotation=45)
    plt.subplot(2, 1, 2)
    width = 0.35
    x = range(len(results_df['run_name']))
    plt.bar([p - width/2 for p in x], results_df['final_gen_loss'], width, label='Generator Loss')
    plt.bar([p + width/2 for p in x], results_df['final_disc_loss'], width, label='Discriminator Loss')
    plt.ylabel('Final Loss')
    plt.xlabel('Model Configuration')
    plt.title('Final Generator and Discriminator Loss by Model Configuration')
    plt.xticks(x, results_df['run_name'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'grid_search_comparison.png'))
    plt.close()
    # Creating combined GIF of best model progress
    best_run_dir = os.path.join(OUTPUT_DIR, best_model['run_name'])
    image_files = sorted([os.path.join(best_run_dir, f) for f in os.listdir(best_run_dir) if f.startswith('epoch_') and f.endswith('.png')])
    images = [imageio.imread(f) for f in image_files]
    imageio.mimsave(GIF_PATH, images, fps=5)
    return best_model