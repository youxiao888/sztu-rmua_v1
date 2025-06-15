import onnx

# 加载模型
model = onnx.load("/home/amov/Downloads/yolo11n-lamp-10.0x-finetune.onnx")

# 将IR版本从10降至9
if model.ir_version > 9:
    model.ir_version = 9

# 保存降级后的模型
onnx.save(model, "/home/amov/Downloads/yolo11n-lamp-10.0x-finetune_ir9.onnx")