import sys
try:
    # Python 3.8+ 推荐
    from importlib.metadata import distributions
except ImportError:
    # 兼容低版本 Python
    from pkg_resources import working_set as distributions


def get_python_version() -> str:
    """获取当前Python版本"""
    return sys.version.split()[0]


def get_all_installed_packages() -> list:
    """获取所有已安装依赖包名称+版本"""
    pkg_list = []
    for dist in distributions():
        pkg_name = dist.metadata["Name"]
        pkg_ver = dist.version
        pkg_list.append((pkg_name, pkg_ver))
    # 按包名排序
    pkg_list.sort(key=lambda x: x[0].lower())
    return pkg_list


def export_to_requirements(file_path: str = "requirements.txt"):
    """导出为标准 requirements.txt"""
    pkgs = get_all_installed_packages()
    with open(file_path, "w", encoding="utf-8") as f:
        for name, ver in pkgs:
            f.write(f"{name}=={ver}\n")
    print(f"依赖列表已导出至: {file_path}")


if __name__ == "__main__":
    # 1. 打印 Python 版本
    py_ver = get_python_version()
    print(f"当前 Python 版本: {py_ver}\n")

    # 2. 打印所有依赖包
    print("=== 已安装依赖包及版本 ===")
    packages = get_all_installed_packages()
    for name, ver in packages:
        print(f"{name:25} == {ver}")

    # 3. 导出到 requirements.txt（可选）
    export_to_requirements()