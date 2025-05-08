from config import DATA_DIR, GIF_PATH, RESULTS_DIR
from data_utils import load_images, prepare_datasets
from grid_search import grid_search
import os

if __name__ == "__main__":
    all_images = load_images() #loading and preprocessing of images
    train_dataset, val_images_tf, test_images_tf = prepare_datasets(all_images) #Spliting the dataset
    train_images, val_images, test_images = all_images[:int(0.9 * len(all_images))], all_images[int(0.9 * len(all_images)):int(0.99 * len(all_images))], all_images[int(0.99 * len(all_images)):]
    best_model = grid_search(train_dataset, val_images_tf, test_images_tf, train_images, val_images, test_images) #hyperparameter tuning using grid search
    print("\n===== Training Complete =====")
    print(f"Best model: {best_model['run_name']}")
    print(f"Parameters:")
    print(f"  - Latent Dimension: {best_model['latent_dim']}")
    print(f"  - Generator Filters: {best_model['generator_filters']}")
    print(f"  - Dropout Rate: {best_model['dropout_rate']}")
    print(f"  - Learning Rate: {best_model['learning_rate']}")
    print(f"Performance:")
    print(f"  - Test FID: {best_model['test_fid']}")
    print(f"  - Final Generator Loss: {best_model['final_gen_loss']}")
    print(f"  - Final Discriminator Loss: {best_model['final_disc_loss']}")
    print(f"\nTraining progress GIF saved to: {GIF_PATH}")
    print(f"Detailed results saved to: {os.path.join(RESULTS_DIR, 'grid_search_results.csv')}")