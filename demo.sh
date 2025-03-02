# model :DANet , dataset :parking,weather,solar,solar_h,electricity

#parking
python3 run.py --model_id DANet_h --model DANet --batch_size 128 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'parking_11' --data_path './data/parking_data_h_new.csv' --seq_number 47  --test_index 47 --end_index 53 --pred_len 720 --label_len 720
python3 run.py --model_id DANet_h --model DANet --batch_size 128 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'parking_12' --data_path './data/parking_data_h_new.csv' --seq_number 47  --test_index 47 --end_index 53 --pred_len 480 --label_len 480
python3 run.py --model_id DANet_h --model DANet --batch_size 128 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'parking_13' --data_path './data/parking_data_h_new.csv' --seq_number 47  --test_index 47 --end_index 53 --pred_len 168 --label_len 168
python3 run.py --model_id DANet_h --model DANet --batch_size 128 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'parking_14' --data_path './data/parking_data_h_new.csv' --seq_number 47  --test_index 47 --end_index 53 --pred_len 360 --label_len 360

#solar
python3 run.py --model_id DANet_h --model DANet --batch_size 64 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'solar_1' --data_path './data/solar.csv' --seq_number 110  --test_index 110 --end_index 137 --pred_len 720 --label_len 720
python3 run.py --model_id DANet_h --model DANet --batch_size 64 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'solar_2' --data_path './data/solar.csv' --seq_number 110  --test_index 110 --end_index 137 --pred_len 480 --label_len 480
python3 run.py --model_id DANet_h --model DANet --batch_size 64 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'solar_3' --data_path './data/solar.csv' --seq_number 110  --test_index 110 --end_index 137 --pred_len 168 --label_len 168
python3 run.py --model_id DANet_h --model DANet --batch_size 64 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'solar_4' --data_path './data/solar.csv' --seq_number 110  --test_index 110 --end_index 137 --pred_len 360 --label_len 360

#weather
py4thon3 run.py --model_id DANet_h --model DANet --batch_size 128 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'weather_1' --data_path './data/weather.csv' --seq_number 35 --test_index 35 --end_index 41  --pred_len 720 --label_len 720
python3 run.py --model_id DANet_h --model DANet --batch_size 128 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'weather_2' --data_path './data/weather.csv' --seq_number 35 --test_index 35 --end_index 41  --pred_len 480 --label_len 480
python3 run.py --model_id DANet_h --model DANet --batch_size 128 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'weather_3' --data_path './data/weather.csv' --seq_number 35 --test_index 35 --end_index 41  --pred_len 168 --label_len 168
python3 run.py --model_id DANet_h --model DANet --batch_size 128 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'weather_4' --data_path './data/weather.csv' --seq_number 35 --test_index 35 --end_index 41  --pred_len 360 --label_len 360

#solar_H
python3 run.py --model_id DANet_h --model DANet --batch_size 64 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'solarh_1' --data_path './data/solar_h.csv' --seq_number 110  --test_index 110 --end_index 137  --pred_len 720 --label_len 720
python3 run.py --model_id DANet_h --model DANet --batch_size 64 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'solarh_2' --data_path './data/solar_h.csv' --seq_number 110  --test_index 110 --end_index 137  --pred_len 480 --label_len 480
python3 run.py --model_id DANet_h --model DANet --batch_size 64 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'solarh_3' --data_path './data/solar_h.csv' --seq_number 110  --test_index 110 --end_index 137  --pred_len 168 --label_len 168
python3 run.py --model_id DANet_h --model DANet --batch_size 64 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'solarh_4' --data_path './data/solar_h.csv' --seq_number 110  --test_index 110 --end_index 137  --pred_len 360 --label_len 360

#electricity
python3 run.py --model_id DANet_h --model DANet --batch_size 32 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'electricity_1' --data_path './data/electricity_h.csv' --seq_number 320 --test_index 320 --end_index 370 --pred_len 720 --label_len 720
python3 run.py --model_id DANet_h --model DANet --batch_size 32 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'electricity_2' --data_path './data/electricity_h.csv' --seq_number 320 --test_index 320 --end_index 370 --pred_len 480 --label_len 480
python3 run.py --model_id DANet_h --model DANet --batch_size 32 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'electricity_3' --data_path './data/electricity_h.csv' --seq_number 320 --test_index 320 --end_index 370 --pred_len 168 --label_len 168
python3 run.py --model_id DANet_h --model DANet --batch_size 32 --train_epochs 300 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'electricity_4' --data_path './data/electricity_h.csv' --seq_number 320 --test_index 320 --end_index 370 --pred_len 360 --label_len 360

#traffic
python3 run.py --model_id DANet_h --model DANet --batch_size 16 --train_epochs 100 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'dline_traffic_1' --data_path './data/PeMS.csv' --seq_number 750 --test_index 750 --end_index 862 --pred_len 720 --label_len 720
python3 run.py --model_id DANet_h --model DANet --batch_size 16 --train_epochs 100 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'dline_traffic_2' --data_path './data/PeMS.csv' --seq_number 750 --test_index 750 --end_index 862 --pred_len 480 --label_len 480
python3 run.py --model_id DANet_h --model DANet --batch_size 16 --train_epochs 100 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'dline_traffic_3' --data_path './data/PeMS.csv' --seq_number 750 --test_index 750 --end_index 862 --pred_len 168 --label_len 168
python3 run.py --model_id DANet_h --model DANet --batch_size 16 --train_epochs 100 --model_type 3 --train_only --data parking_h_at --isRNN 0 --random_size 1200 --des 'dline_traffic_4' --data_path './data/PeMS.csv' --seq_number 750 --test_index 750 --end_index 862 --pred_len 360 --label_len 360
