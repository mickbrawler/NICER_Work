import json
import argparse

def combine_multiprocessing(label):
    # currently not using label argument. Should use in the future...

    p_densities = {}
    for job in range(100):

        with open("data/multiprocessing/{}.json".format(job),"r") as f:
            piece = json.load(f)

        for key in list(piece.keys()):
            p_densities[key] = piece[key]

    filename = "../development/run_data/GW_EM_reweight_spectral_sampled_p_rho_metrohast.json"
    with open(filename, "w") as f:
        json.dump(p_densities, f, indent=2, sort_keys=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="Filename label", type=str)
    args = parser.parse_args()

    combine_multiprocessing(args.label)

