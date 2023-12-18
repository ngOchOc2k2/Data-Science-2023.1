# Re-ranker


## Các Tệp

1. **bm25.py:** File này giúp load BM25 model.

2. **config.md:** Mô tả các tham số sử dụng.

3. **model**
    - **base_model.py:** File chứa source về Adapter và hàm load, save tham số, model.
    - **bert_model.py:** File chứa source về Bert Model.
    - **cross-encoder.py:** File chứa source re-ranker model.

4. **data_loader.py** File chứa source về Dataloader


5. **eval.py** File chứa source để đánh giá model


6. **main.py** File chạy train model


7. **loss.py** File chứa hàm loss

8. **train_v2.py** File chứa hàm train model

## Cách Sử Dụng

**Chạy file: python main.py**

## Yêu Cầu

- pytorch >= 2.0
- transformer >= 4.2

