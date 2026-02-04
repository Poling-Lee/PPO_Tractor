from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner

if __name__ == '__main__':
    config = {
        'replay_buffer_size': 50000,
        'replay_buffer_episode': 400,
        'model_pool_size': 20,
        'model_pool_name': 'model-pool',
        'num_actors': 4,
        'episodes_per_actor': 2000,
        'gamma': 0.98,
        'lambda': 0.95,
        'min_sample': 200,
        'batch_size': 256,
        'epochs': 5,
        'clip': 0.2,
        'lr': 3e-5,
        'value_coeff': 1,
        'entropy_coeff': 0.01,
        'device': 'cpu',
        'ckpt_save_interval': 300,
        'process_reward_weight': 0.6,    # 过程奖励的权重
        'team_coop_weight': 0.3,         # 团队协作奖励权重
        'defensive_weight': 0.4,         # 防守奖励权重
        
        # 新增：惩罚相关参数
        'punish_decay': 0.95,            # 惩罚衰减因子
        'team_punish_ratio': 0.25,       # 团队惩罚比例
        
        'ckpt_save_path': 'checkpoint/'
    }
    
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actors.append(actor)
    learner = Learner(config, replay_buffer)
    
    for actor in actors: actor.start()
    learner.start()
    
    for actor in actors: actor.join()
    learner.terminate()