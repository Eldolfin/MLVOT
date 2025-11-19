{
  inputs = {
    utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, utils }: utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      opencvGtk = pkgs.opencv4.override (old : { enableGtk2 = true; });
      opencv-pythonGtk = pkgs.python313Packages.opencv-python.override (old : {opencv4 = opencvGtk;});
    in
    {
      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [
          (python313.withPackages (ppkgs: with ppkgs; [
            opencv-pythonGtk
            ipython
            ruff
            ty
            python-lsp-server
          ]))
        ];
      };
    }
  );
}
