import coverage
import unittest
import os
import sys
import shutil

def clean_directories():
    """清理所有缓存和临时目录"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 清理__pycache__目录
    for root, dirs, files in os.walk(current_dir):
        if '__pycache__' in dirs:
            pycache_dir = os.path.join(root, '__pycache__')
            print(f'清理缓存目录: {pycache_dir}')
            shutil.rmtree(pycache_dir)
    
    # 清理coverage数据
    coverage_data = os.path.join(current_dir, '.coverage')
    if os.path.exists(coverage_data):
        os.remove(coverage_data)
    
    # 清理coverage_html目录
    html_dir = os.path.join(current_dir, 'coverage_html')
    if os.path.exists(html_dir):
        shutil.rmtree(html_dir)

def get_source_files():
    """获取所有需要测试的源文件"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    source_files = []
    
    # 主应用文件
    source_files.append(os.path.join(current_dir, 'app.py'))
    source_files.append(os.path.join(current_dir, 'config.py'))
    
    # 模块文件
    modules = ['dehazing', 'detect', 'fusion']
    for module in modules:
        module_dir = os.path.join(current_dir, module)
        if os.path.exists(module_dir):
            for file in os.listdir(module_dir):
                if file.endswith('.py'):
                    source_files.append(os.path.join(module_dir, file))
    
    return source_files

def run_tests():
    """运行测试并收集覆盖率数据"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 清理环境
    clean_directories()
    
    # 获取源文件
    source_files = get_source_files()
    print('\n要测试的源文件:')
    for file in source_files:
        print(f'  {os.path.relpath(file, current_dir)}')
    
    # 创建覆盖率对象
    cov = coverage.Coverage(
        branch=True,
        source=[current_dir],
        omit=[
            '*/__pycache__/*',
            '*/__tests__/*',
            '*/venv/*',
            '*/env/*',
            '*/site-packages/*',
            'run_coverage.py',
            '*/coverage_html/*'
        ]
    )
    
    # 开始收集覆盖率数据
    cov.start()
    
    # 运行测试
    test_dir = os.path.join(current_dir, '__tests__')
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 停止收集覆盖率数据
    cov.stop()
    cov.save()
    
    # 生成报告
    print('\n测试结果:')
    print(f'运行测试数: {result.testsRun}')
    print(f'成功: {result.testsRun - len(result.failures) - len(result.errors)}')
    print(f'失败: {len(result.failures)}')
    print(f'错误: {len(result.errors)}')
    
    print('\n覆盖率报告:')
    try:
        # 生成控制台报告
        cov.report()
        
        # 生成HTML报告
        html_dir = os.path.join(current_dir, 'coverage_html')
        os.makedirs(html_dir, exist_ok=True)
        cov.html_report(directory=html_dir)
        print(f'\nHTML报告已生成在 {html_dir} 目录中')
        
        # 生成XML报告
        xml_file = os.path.join(current_dir, 'coverage.xml')
        cov.xml_report(outfile=xml_file)
        print(f'XML报告已生成: {xml_file}')
        
    except Exception as e:
        print(f'生成报告时出错: {str(e)}')
        print('\n当前目录结构:')
        for root, dirs, files in os.walk(current_dir):
            level = root.replace(current_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                if f.endswith('.py'):
                    print(f'{subindent}{f}')

if __name__ == '__main__':
    run_tests() 