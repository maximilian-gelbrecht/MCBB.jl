using Documenter, HighBifLib, DifferentialEquations

makedocs(sitename="HighBifLib", modules=[HighBifLib], doctest=true,
pages = [
    "Home" => "index.md",
    "Manual" => Any[
        "Basic Usage" => "basic_usage.md",
        "introspective_features.md", "multidim_parameters.md"
        ],
    "Reference" => Any[
        "Problem Types" => "ref/problem_types.md",
        "ParameterVar" => "ref/parameter_var.md",
        "Evaluation Functions" =>"ref/evaluation_funcs.md",
        ]
    ],
    )
