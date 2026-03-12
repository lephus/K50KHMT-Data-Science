K50KHMT-Data-Science

### Product Recommendation System Overview

This project includes a **product recommendation system** that predicts which products a customer is most likely to be interested in, based on their past purchase history. The problem is framed as: for each customer, find the **Top-N products** they have not purchased yet, using both their own behavior and the behavior of similar customers.

### Dataset Design

The data is organized into five CSV files that mirror a real-world sales database schema:

- **`products.csv`**: 140 products with fields such as product ID, name, category (e.g., Ao dai, suit, dress), price, material, and inventory.
- **`customers.csv`**: 500 customers with ID, name, email, preferred categories, budget level (low/medium/high), and city.
- **`orders.csv`**: 3,872 orders with order ID, customer ID, order date, status (completed/cancelled), and total amount.
- **`order_items.csv`**: 9,624 rows of order line items including product ID, quantity, unit price, and rating (1–5).
- **`user_item_interactions.csv`**: 4,114 (customer, product) pairs with aggregated interaction statistics.

The **interaction score** for each (customer, product) pair is computed as:

\[
\text{score} = 0.5 \cdot \log(1 + \text{purchase\_count}) + 0.5 \cdot \text{avg\_rating}
\]

This balances **purchase frequency** and **post-purchase satisfaction**. The data is highly sparse (about **94%** of all possible (customer, product) pairs have no interactions), which is typical for recommendation problems.

### Model: Matrix Factorization with SGD

The core model is a **Matrix Factorization** approach optimized using **Stochastic Gradient Descent (SGD)**, implemented from scratch (no ML libraries).

Each customer and product is represented by a vector of **latent factors**:

- **User factors `U`**: shape `(n_users, n_factors)` — learned preference vectors.
- **Item factors `V`**: shape `(n_items, n_factors)` — learned product attribute vectors.
- **User bias `b_u`** and **item bias `b_i`**: capture systematic tendencies of specific users or items.
- **Global bias `b_global`**: overall average interaction score.

The predicted score for user \(u\) and item \(i\) is:

\[
\hat{r}_{u,i} = b_{\text{global}} + b_u[u] + b_i[i] + U[u] \cdot V[i]
\]

SGD iteratively updates `U`, `V`, `b_u`, and `b_i` for each observed interaction to minimize squared error with L2 regularization. Key hyperparameters:

- **`n_factors = 20`**: number of latent dimensions.
- **`lr = 0.005`**: learning rate.
- **`reg = 0.02`**: regularization strength.
- **`n_epochs = 30`**: number of passes over the training data.

### Evaluation

The 4,114 interactions are split into **train/validation/test** with a 70/15/15 ratio:

- **Train**: used to update model parameters.
- **Validation**: used to monitor training and detect overfitting.
- **Test**: held-out set for final evaluation.

Two groups of metrics are used:

- **Regression metrics** (for score prediction):
  - **MSE ≈ 0.18**
  - **RMSE ≈ 0.43**
  - **MAE ≈ 0.34**
  - **R² < 0** (common with very sparse data; baseline “predict the mean” is hard to beat in global variance terms).

- **Ranking metrics** (for recommendation quality):
  - **Precision@K**: fraction of recommended items that are actually relevant.
  - **Recall@K**: fraction of relevant items that are recovered in the Top-K recommendations.
  - **NDCG@K**: measures both correctness and position of relevant items in the ranked list.
  - **Coverage@10 ≈ 50%**: about half of all products are recommended at least once, which avoids recommending only a small set of popular items.

Training and validation RMSE show a small gap, indicating that **overfitting is controlled** by regularization.

### End-to-End Pipeline

1. **Data collection** from the transactional database into CSV files.
2. **Interaction matrix construction** using purchase counts and ratings.
3. **Train/val/test split** with a fixed random seed.
4. **Model initialization** for user/item factors and biases.
5. **Training with SGD** for a fixed number of epochs.
6. **Evaluation** with regression and ranking metrics on the test set.
7. **Recommendation generation**: for each user, score all unseen items, sort by predicted score, and return the Top-N items.

### Future Improvements

Potential directions to enhance the system include:

- Connecting directly to the **production database** for real transaction data.
- Adding **content-based features** (product metadata) for a hybrid recommender.
- Using **early stopping** based on validation RMSE instead of a fixed number of epochs.
- Incorporating **implicit feedback** (views, clicks, dwell time).
- Trying **ALS (Alternating Least Squares)** as an alternative optimization method.
- Applying **business rules and re-ranking** based on budget, stock levels, and campaign priorities.