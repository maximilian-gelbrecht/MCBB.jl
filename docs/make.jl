using Documenter, MCBB, DifferentialEquations, Clustering

makedocs(sitename="MCBB", modules=[MCBB], doctest=true,
pages = [
    "Home" => "index.md",
    "Manual" => Any[
        "Basic Usage" => "basic_usage.md",
        "introspective_features.md", "hpc.md",
        "multidim_parameters.md", "custom_problems.md",
        "continuation.md"
        ],
    "Reference" => Any[
        "Problem Types" => "ref/problem_types.md",
        "ParameterVar" => "ref/parameter_var.md",
        "Evaluation Functions" =>"ref/evaluation_funcs.md",
        "Clustering Functions" =>"ref/clustering_funcs.md",
        "Continuation" => "ref/continuation.md",
        "Systems" => "ref/systems.md",
        "Custom Problems" => "ref/custom_problem.md"
        ]
    ]
    )
