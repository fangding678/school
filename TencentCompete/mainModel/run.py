import subprocess

if __name__ == "__main__":
    cmd1 = "nohup python /home/ubuntu/tecent/mainModel/main.py >rlog.txt 2>&1 &"
    cmd2 = "triple_format_to_libfm.pl -in all_simple_fm_no_user.csv -target 28 -header 1 -separator ','"
    cmd3 = "nohup libFM -task c -train train_fm -test test_fm -dim '1,1,24' -out 'result_24.csv' -rlog 'log24.txt' >nohup24.out 2>&1 &"
    subprocess.call(cmd1, shell=True)
