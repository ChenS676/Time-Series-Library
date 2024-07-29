print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

# 打印路径进行调试
print(f"Root path: {args.root_path}")
print(f"Data path: {args.data_path}")

try:
    exp.train(setting)
except Exception as e:
    print(f"Error during training: {e}")

print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
exp.test(setting)
torch.cuda.empty_cache()
