import os
import subprocess
import gradio as gr

def process_video(video_file, text, output_dir):
    """处理视频文件"""
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 第一阶段：识别视频内容
        cmd = [
            "python", "funclip/videoclipper.py",
            "--stage", "1",
            "--file", video_file,
            "--output_dir", output_dir
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"第一阶段处理失败: {result.stderr}"
        
        # 第二阶段：剪辑视频
        cmd = [
            "python", "funclip/videoclipper.py",
            "--stage", "2",
            "--file", video_file,
            "--dest_text", text,
            "--output_dir", output_dir
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"第二阶段处理失败: {result.stderr}"
        
        # 获取输出文件路径
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_clip_no0.mp4")
        
        return output_file
    except Exception as e:
        return f"处理失败: {str(e)}"

def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 视频剪辑工具")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="上传视频")
                text_input = gr.Textbox(label="输入要剪辑的文本")
                with gr.Row():
                    output_dir = gr.Textbox(
                        label="输出目录", 
                        value="./output", 
                        info="指定输出文件的保存目录",
                        placeholder="例如：E:/output 或 ./output"
                    )
                    process_btn = gr.Button("开始处理", variant="primary")
            
            with gr.Column():
                output_video = gr.Video(label="处理结果")
                status = gr.Textbox(label="状态信息", lines=3)
        
        def process(video, text, output_dir):
            if not video or not text:
                return None, "请上传视频并输入要剪辑的文本"
            
            # 确保输出目录是绝对路径
            output_dir = os.path.abspath(output_dir)
            
            # 处理视频
            result = process_video(video, text, output_dir)
            if result.startswith("处理失败") or result.startswith("第一阶段处理失败") or result.startswith("第二阶段处理失败"):
                return None, result
            
            return result, f"处理完成！\n输出文件保存在：{result}"
        
        process_btn.click(
            fn=process,
            inputs=[video_input, text_input, output_dir],
            outputs=[output_video, status]
        )
    
    return demo 