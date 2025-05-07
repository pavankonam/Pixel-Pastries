# 🍰 Pixel Pastries: GAN-based Dessert Dream Machine

Welcome to **Pixel Pastries**, a generative deep learning project that creates fantasy dessert images using a custom-built Generative Adversarial Network (GAN). This project focuses on generating **original**, **aesthetically pleasing**, and **realistic** desserts—ranging from cakes and cookies to donuts and beyond.

## 📸 Dataset

We use a **filtered subset of the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)**, selecting only dessert-related classes (e.g., cakes, cookies, donuts, brownies). The goal is to focus purely on dessert imagery for training.

To create the custom dessert dataset, run:

```bash
python dataset_creation.py

This script filters the raw Food-101 dataset and organizes it into train/, val/, and test/ folders for use during training and evaluation.

