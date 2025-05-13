sweep_config = {
            'method': 'random',  
            'metric': {
                'name': 'time',  
                'goal': 'minimize' 
            },
            'parameters': {
                'learning_rate': {
                    'distribution':'log_uniform_values',
                    'min': 1e-5,
                    'max': 1e-4                
                },
                'batch_size': {
                    'values': [8, 16]
                    # 'distribution':'q_uniform',
                    # 'q': 2,
                    # 'min': 8,
                    # 'max': 16,
                },
                'optimizer_type': {
                    'values': ['SGD', 'Adam', 'AdamW']
                },
                # 'P': {  
                #     # 'values': [5, 7, 10, 15]
                #     'distribution':'q_uniform',
                #     'q': 1,
                #     'min': 2,
                #     'max': 15,                
                # },
                'threshold': {              
                    'min': 1e-6,
                    'max': 1e-1
                },
                'ema_decay': {
                    'min': 0.01,
                    'max': 0.99
                }
                # 'adaptivity_type': {
                #     'values': ['mean', 'median']
                # },
                # 'model_name': {
                #     'values': ['cnn', 'resnet18', 'mobilenet_v2']
                # },
                # 'training_mode': {
                #     'values': ['parallel', 'serial']
                # }
            }
        }