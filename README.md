# TCD Official Repository

This is the official repository for the paper "TCD". Below is the structure of the repository and instructions on how to reproduce the experiments.

## Repository Structure
```
.
├── README.md
├── TCDTrainer.py
├── clone_dataset.py
├── dataset
│   ├── fun_test.pkl
│   ├── fun_train.pkl
│   ├── gcj_all.pkl
│   ├── java_fun_test.pkl
│   ├── java_fun_train.pkl
│   ├── java_random_test.pkl
│   ├── java_random_train.pkl
│   ├── random_test.pkl
│   └── random_train.pkl
├── freeze_param.py
├── train_model.py 
└── vis
    ├── cl_embedding.pt
    ├── cl_labels.npy
    ├── tcd_embedding.pt
    ├── tcd_labels.npy
    ├── vis.ipynb
    └── vis.pdf
```

## Reproducing Experiments

To reproduce the experiments, follow these steps:

1. **Clone the repository:**

    ```sh
    git clone https://github.com/tcdsaner/TCD
    cd TCD
    ```

2. **Install the required dependencies:**

    Make sure you have Python 3.10.14 installed. Then, install the required Python packages:

    ```
    pip install transformers==4.44.2
    pip install torch==2.4.0
    pip install tokenizers=0.19.1
    pip install pandas==2.2.2
    pip install scikit-learn
    ```

3. **Prepare the dataset:**

    The dataset files are located in the `dataset/` directory. You can load the dataset using the `clone_dataset.py` script.

    ```sh
    python clone_dataset.py
    ```

4. **Train the model:**

    Use the `train_model.py` script to train the model. You can adjust the training parameters by modifying the script or passing arguments.

    ```sh
    python train_model.py --epochs 4 --lr 2e-5 --batch-size-per-replica 8 --train_data ./dataset/fun_train.pkl --test_data ./dataset/fun_test.pkl
    ```

5. **Visualize the results:**

    The visualization scripts are located in the `vis/` directory. You can use the Jupyter notebook `vis.ipynb` to visualize the embeddings.

## Contact

For any questions or issues, please open an issue in this repository or contact the authors of the paper.
