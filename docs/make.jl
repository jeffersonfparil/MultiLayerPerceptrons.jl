using MultiLayerPerceptrons
using Documenter

DocMeta.setdocmeta!(
    MultiLayerPerceptrons,
    :DocTestSetup,
    :(using MultiLayerPerceptrons);
    recursive=true,
)

makedocs(;
    modules=[MultiLayerPerceptrons],
    authors="jeffersonparil@gmail.com",
    sitename="MultiLayerPerceptrons.jl",
    format=Documenter.HTML(;
        canonical="https://jeffersonfparil.github.io/MultiLayerPerceptrons.jl",
        edit_link="main",
        assets=String[],
        size_threshold=1000000,
    ),
    pages=["Home" => "index.md"],
)

deploydocs(;
    repo="github.com/jeffersonfparil/MultiLayerPerceptrons.jl",
    devbranch="main",
)
