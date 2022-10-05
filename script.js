// This JSON will be in a different file
// var paper = {
//     "title": "PBNS: Physically Based Neural Simulation for Unsupervised Outfit Pose Space Deformation",
//     "conference": "SIGGRAPH Asia 2021",
//     "authors": [
//         {
//             "name": "Hugo Bertiche",
//             "email": "hugo_bertiche@hotmail.com",
//             "affiliations": [1, 2]
//         },
//         {
//             "name": "Meysam Madadi",
//             "email": "mmadadi@cvc.uab.cat",
//             "affiliations": [1, 2]
//         },
//         {
//             "name": "Sergio Escalera",
//             "email": "sescalera@ub.edu",
//             "affiliations": [1, 2]
//         }
//     ],
//     "affiliations": ["Universitat de Barcelona", "Computer Vision Center"],
//     "URLs": {
//         "paper": "https://dl.acm.org/doi/10.1145/3478513.3480479",
//         "arxiv": "https://arxiv.org/abs/2012.11310",
//         "youtube": "https://youtu.be/ALwhjm40zRg",
//         "github": "https://github.com/hbertiche/PBNS",
//         "data": null
//     },
//     "abstract": "We present a methodology to automatically obtain Pose Space Deformation (PSD) basis for rigged garments through deep learning. Classical approaches rely on Physically Based Simulations (PBS) to animate clothes. These are general solutions that, given a sufficiently fine-grained discretization of space and time, can achieve highly realistic results. However, they are computationally expensive and any scene modification prompts the need of re-simulation. Linear Blend Skinning (LBS) with PSD offers a lightweight alternative to PBS, though, it needs huge volumes of data to learn proper PSD. We propose using deep learning, formulated as an implicit PBS, to un-supervisedly learn realistic cloth Pose Space Deformations in a constrained scenario: dressed humans. Furthermore, we show it is possible to train these models in an amount of time comparable to a PBS of a few sequences. To the best of our knowledge, we are the first to propose a neural simulator for cloth. While deep-based approaches in the domain are becoming a trend, these are data-hungry models. Moreover, authors often propose complex formulations to better learn wrinkles from PBS data. Supervised learning leads to physically inconsistent predictions that require collision solving to be used. Also, dependency on PBS data limits the scalability of these solutions, while their formulation hinders its applicability and compatibility. By proposing an unsupervised methodology to learn PSD for LBS models (3D animation standard), we overcome both of these drawbacks. Results obtained show cloth-consistency in the animated garments and meaningful pose-dependant folds and wrinkles. Our solution is extremely efficient, handles multiple layers of cloth, allows unsupervised outfit resizing and can be easily applied to any custom 3D avatar."
// }

get = id => document.getElementById(id);

function author_node(author) {
    var span = document.createElement("span");
    var a = document.createElement("a");
    var sup = document.createElement("sup");
    a.textContent = author.name;
    a.href = "mailto:" + author.email;
    sup.textContent = author.affiliations.map(String).join(",");
    span.appendChild(a);
    span.appendChild(sup);
    return span
}

function affiliations_node(affiliations) {
    var span = document.createElement("span");
    span.innerHTML = affiliations.map((affiliation, index) => 
        "<sup>" + (index + 1).toString() + "</sup>" + affiliation
    ).join(", ");
    return span
}

function copy_bibtex() {
    var range = document.createRange();
    range.selectNode(get("bibtex"));
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
    document.execCommand("copy");
    window.getSelection().removeAllRanges();
}

// Read JSON
fetch("./paper.json").then(response => response.json()).then(response => console.log(response))

// Set paper metadata
get("title").textContent = paper.title;
get("conference").textContent = paper.conference;
paper.authors.map((author, index) => {
    node = author_node(author);
    get("author-list").appendChild(node);
    if(index == paper.authors.length - 1) return;
    node.innerHTML += ", "
})
get("affiliation-list").appendChild(affiliations_node(paper.affiliations));
get("abstract").textContent = paper.abstract;
for(var button in paper.URLs) {
    node = get(button);
    url = paper.URLs[button];
    if(url == null) node.remove();
    else node.href = url;
}
get("video").src = paper.URLs.youtube;
get("copy-button").onclick = copy_bibtex