## Configurations of training a News Recommendations Model
- **news_batch_size**
  - **Description**: Batch size for news embedding.
  - **Suggested Values**: Typically values range from 32 to 512, depending on memory and computational capacity.

- **user_batch_size**
  - **Description**: Batch size for user embedding.
  - **Suggested Values**: Typically values range from 32 to 512, similar to `news_batch_size`.

- **fast_evaluation**
  - **Description**: Whether to use fast evaluation mode.
  - **Suggested Values**: `True` or `False`. Use `True` for quick testing and `False` for detailed evaluation.

- **loss**
  - **Description**: Loss function used in the model.
  - **Suggested Values**: `cross_entropy`, `mean_squared_error`, `hinge_loss`.

- **text_feature**
  - **Description**: Specifies the text features to be used, such as title, abstract, body.
  - **Suggested Values**: `"title"`, `"abstract"`, `"body"`, or a combination such as `"[title,body]"`.

- **cat_feature**
  - **Description**: Specifies the categorical features to be used, such as category and subvert.
  - **Suggested Values**: `"category"`, `"subvert"`, or a combination such as `"[category,subvert]"`.

- **entity_feature**
  - **Description**: Specifies the entity features to be used.
  - **Suggested Values**: `"entity"`, `"abstract"`, or a combination such as `"[entity,abstract]"`.

- **category_dim**
  - **Description**: Dimension for category embedding.
  - **Suggested Values**: Typically between 50 and 200, depending on the complexity of categories.

- **attention_hidden_dim**
  - **Description**: Hidden dimension for the attention mechanism.
  - **Suggested Values**: Typically between 100 and 300. Higher values might capture more detailed relationships.

- **dropout_we**
  - **Description**: Dropout rate for word embedding layer.
  - **Suggested Values**: Between 0 and 1, with a common default of 0.2 to prevent overfitting.

- **dropout_ne**
  - **Description**: Dropout rate for news encoder layer.
  - **Suggested Values**: Between 0 and 1, commonly 0.2.

- **dropout_ce**
  - **Description**: Dropout rate for category encoder layer.
  - **Suggested Values**: Between 0 and 1, commonly 0.2.