sweep_config = {
            'method': 'grid',  
            'metric': {
                'name': 'time',  
                'goal': 'minimize' 
            },
            'parameters': {
                'P': {  
                    # 'values': [5, 7, 10, 15]
                    'distribution':'q_uniform',
                    'q': 1,
                    'min': 1,
                    'max': 101,                
                },
                'device_count':{
                    'distribution':'q_uniform',
                    'q': 1,
                    'min': 1,
                    'max': 8
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