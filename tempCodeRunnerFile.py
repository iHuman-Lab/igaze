with skip_run("run", "fixation_extraction") as check, check():
    raw_fixations, fixation_summary = extract_fixations_from_config(config_path)
    print("Fixation summary:")
    print(fixation_summary)