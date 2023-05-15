import os 
import pandas as pd
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

working_dir = os.path.abspath(os.path.dirname('.'))
data_path = os.path.join(working_dir, "data")
walk = os.walk(data_path)
instances = []
for path, dir_list, file_list in walk:
    for file in file_list:
        if 'txt' in file:
            file_path = os.path.join(working_dir, path, file)
            label = os.path.dirname(file_path).split('/')[-1]
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    text = ' '.join(f.readlines())
                except:
                    print(f"error occurs for {file_path}")
            instances.append((text, label))  

texts = [_[0] for _ in instances]
labels = [_[1] for _ in instances]
df = pd.DataFrame({'text': texts, 'label': labels})
df.to_csv('dataset.csv', index = False)

logger.info(f'The number of samples: {len(instances)}.')