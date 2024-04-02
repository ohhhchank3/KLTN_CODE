import os
import re
import subprocess


def get_latest_tag():
    output = subprocess.check_output(['git', 'tag'])
    tags = output.decode('utf-8').split('\n')[:-1]
    latest_tag = sorted(tags, key=lambda t: tuple(map(int, re.match(r'v(\d+)\.(\d+)\.(\d+)', t).groups())))[-1]
    return latest_tag

def update_version_number(latest_tag, increment):
    major, minor, patch = map(int, re.match(r'v(\d+)\.(\d+)\.(\d+)', latest_tag).groups())
    if increment == 'X':
        major += 1
        minor, patch = 0, 0
    elif increment == 'Y':
        minor += 1
        patch = 0
    elif increment == 'Z':
        patch += 1
    new_version = f"v{major}.{minor}.{patch}"
    return new_version

def main():
    print("Lấy thông tin phiên bản Git gần nhất:")
    latest_tag = get_latest_tag()
    print(latest_tag)

    print("Chọn phần số phiên bản để tăng (X, Y, Z):")
    increment = input().upper()

    while increment not in ['X', 'Y', 'Z']:
        print("Nhập không đúng, vui lòng nhập X, Y hoặc Z:")
        increment = input().upper()

    new_version = update_version_number(latest_tag, increment)
    print(f"Phiên bản mới là: {new_version}")

    print("Xác nhận cập nhật phiên bản và đẩy lên kho lưu trữ từ xa? (y/n)")
    confirmation = input().lower()

    if confirmation == 'y':
        subprocess.run(['git', 'tag', new_version])
        subprocess.run(['git', 'push', 'origin', new_version])
        print("Đã tạo và đẩy phiên bản mới lên kho lưu trữ từ xa.")
    else:
        print("Đã hủy thao tác.")

if __name__ == '__main__':
    main()
