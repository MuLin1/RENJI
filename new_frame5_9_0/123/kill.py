import argparse
import subprocess
import sys

def force_stop_process(target, is_pid=False):
    """
    强制停止指定的 Windows 进程及其所有子进程。
    
    参数:
        target (str): 进程的 PID 或名称
        is_pid (bool): 标识 target 是否为 PID，默认为 False
    """
    try:
        # 构建 taskkill 命令列表
        # /F: 强制终止进程
        # /T: 终止指定的进程及其启动的所有子进程
        if is_pid:
            # 按进程 PID 停止
            command = ["taskkill", "/F", "/T", "/PID", str(target)]
        else:
            # 按进程名称 (Image Name) 停止
            command = ["taskkill", "/F", "/T", "/IM", str(target)]
        
        # 执行系统命令，捕获输出内容以便进行结果分析
        result = subprocess.run(command, capture_output=True, text=True)
        
        # returncode 为 0 表示命令执行成功
        if result.returncode == 0:
            print(f"成功强制停止进程: {target}")
            # 打印系统返回的详细成功信息
            if result.stdout:
                print(result.stdout.strip())
        else:
            print(f"停止进程失败: {target}")
            # 打印系统返回的错误原因（如找不到进程或拒绝访问）
            if result.stderr:
                print(f"错误详情: {result.stderr.strip()}")
            
    except Exception as e:
        # 捕获执行过程中的任何未预料异常
        print(f"执行过程中发生异常: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 初始化命令行参数解析器
    parser = argparse.ArgumentParser(description="Windows 进程强制停止工具")
    
    # 创建互斥参数组，要求必须且只能提供其中一个参数
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--pid", help="指定要停止的进程 PID (例如: 1234)")
    group.add_argument("-n", "--name", help="指定要停止的进程名称 (例如: chrome.exe)")
    
    # 解析用户输入的参数
    args = parser.parse_args()
    
    # 根据用户提供的参数类型，调用核心函数
    if args.pid:
        force_stop_process(args.pid, is_pid=True)
    elif args.name:
        force_stop_process(args.name, is_pid=False)