import os
import subprocess
from pathlib import Path

SEP_SAMPLES_DIR = Path("separation_samples")

N_ITER = {
    "five": 3,
    "overiva": 10,
    "auxiva": 50,
}

ALGO_NAMES = {
    "five": "FIVE",
    "overiva": "OverIVA",
    "auxiva": "AuxIVA",
}

if __name__ == "__main__":

    if not SEP_SAMPLES_DIR.exists():
        os.mkdir(SEP_SAMPLES_DIR)

    f = open(SEP_SAMPLES_DIR / "table.html", "w")

    print(
        """<table>
  <tr>
    <td># mics</td>
    <td>sample #</td>
    <td>algo.</td>
    <td>clean</td>
    <td>mix</td>
    <td>output</td>
    <td>SDR</td>
    <td>SIR</td>
    <td>iter.</td>
    <td>runtime</td>
  </tr>""",
        file=f,
    )

    for sinr in [5]:
        for dist in ["gauss"]:
            for n_mics in [2, 3, 5, 8]:
                for i_seed, seed in enumerate(["2785643", "398745627", "58984517"]):
                    for algo in ["five", "overiva", "auxiva"]:

                        print(f"sinr={sinr} mics={n_mics} seed={seed} algo={algo}")

                        command = [
                            "python",
                            "./example.py",
                            "-m",
                            str(n_mics),
                            "-a",
                            algo,
                            "-d",
                            "gauss",
                            "-n",
                            str(N_ITER[algo]),
                            "--seed",
                            str(seed),
                            "--save",
                            "--no_cb",
                            "--no_plot",
                        ]

                        out = subprocess.run(command, capture_output=True)

                        if out.returncode != 0:
                            print("Failed!!")
                            print("stderr:")
                            print(out.stderr)
                            print("stdout:")
                            print(out.stdout)

                        else:
                            lines = out.stdout.decode().split("\n")

                            for l in lines:
                                e = l.split()
                                if len(e) == 0:
                                    continue
                                elif l.startswith("Processing"):
                                    runtime = e[2]
                                elif l.startswith("SDR"):
                                    sdr = e[6]
                                elif l.startswith("SIR"):
                                    sir = e[6]

                            print(
                                f"""  <tr>
    <td>{n_mics}</td>
    <td>{i_seed + 1}</td>
    <td>{ALGO_NAMES[algo]}</td>
    <td><audio controls="controls" type="audio/wav" src="<SEPDIR>/sample_{sinr}_{seed}_{algo}_{dist}_{n_mics}_ref.wav"><a>play</a></audio></td>
    <td><audio controls="controls" type="audio/wav" src="<SEPDIR>/sample_{sinr}_{seed}_{algo}_{dist}_{n_mics}_mix.wav"><a>play</a></audio></td>
    <td><audio controls="controls" type="audio/wav" src="<SEPDIR>/sample_{sinr}_{seed}_{algo}_{dist}_{n_mics}_source0.wav"><a>play</a></audio></td>
    <td>{sdr} dB</td>
    <td>{sir} dB</td>
    <td>{N_ITER[algo]}</td>
    <td>{runtime} s</td>
  </tr>""",
                                file=f,
                            )

    print("</table>", file=f)

    f.close()
