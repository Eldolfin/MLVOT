{
  inputs = {
    utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, utils }: utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      # opencvGtk = pkgs.opencv4.override (old : { enableGtk2 = true; });
      # opencv-pythonGtk = pkgs.python313Packages.opencv-python.override (old : {opencv4 = old.opencv4.override (old : { enableGtk3 = true; });});
    in
    {
      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [
          (python313.withPackages (ppkgs: with ppkgs; [
            # opencv-pythonGtk
            opencv-python
            numpy

            # dev tools
            ipython
            ipdb
            ruff
            ty
            python-lsp-server
          ]))
        ];
      };
    }
  );
}
