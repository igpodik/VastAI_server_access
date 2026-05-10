cd ~/avito_cup/data
python popular.py \
    --eval-user-events eval_user_events.pq \
    --item-features item_features.parquet \
    --eval-users eval_users.csv \
    --out sub_popular.csv

