"""
RAG module loader - Handles dynamic loading of RAG modules with versioning support
"""
import importlib.util
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class RAGLoader:
    """Handles loading and management of RAG graph modules"""
    
    def __init__(self):
        self.current_graph = None
        self.current_version = None
        self.load_error = None
    
    def load_module(self, version: str = None) -> Tuple[bool, Optional[str]]:
        """
        Load the RAG module with optional version selection
        Returns success status and error message if applicable
        """
        try:
            # Determine paths to check for rag.py
            paths_to_check = [
                Path(__file__).parent.parent / "rag.py",                 # Adjacent to services
                Path(__file__).parent.parent.parent / "rag.py",          # App root
                Path(__file__).parent.parent / "versions" / f"rag_{version}.py" if version else None,  # Versioned
                Path(__file__).parent.parent.parent.parent / "rag.py"    # Project root
            ]
            
            # Filter out None entries
            paths_to_check = [p for p in paths_to_check if p]
            
            # Try each path
            for rag_path in paths_to_check:
                if rag_path.exists():
                    logger.info(f"Loading RAG module from {rag_path}")
                    
                    # Dynamically load the module
                    module_name = f"rag_{version}" if version else "rag"
                    spec = importlib.util.spec_from_file_location(module_name, rag_path)
                    if spec and spec.loader:
                        rag = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = rag  # Add to sys.modules for proper imports
                        spec.loader.exec_module(rag)
                        
                        # Get the graph object
                        if hasattr(rag, "graph"):
                            # Validate graph has invoke method
                            if not hasattr(rag.graph, "invoke"):
                                self.load_error = f"rag.graph found in {rag_path} but has no invoke method"
                                logger.error(self.load_error)
                                return False, self.load_error
                                
                            # Store the loaded graph
                            self.current_graph = rag.graph
                            self.current_version = version
                            self.load_error = None
                            logger.info(f"Successfully loaded RAG graph from {rag_path}")
                            return True, None
                        else:
                            logger.error(f"No 'graph' object found in {rag_path}")
                    else:
                        logger.error(f"Failed to load module spec from {rag_path}")
            
            # If we get here, we didn't find a valid rag.py
            self.load_error = "Could not find rag.py in any expected locations"
            logger.error(self.load_error)
            return False, self.load_error
            
        except Exception as e:
            self.load_error = f"Error loading RAG module: {str(e)}"
            logger.error(self.load_error)
            logger.error(traceback.format_exc())
            return False, self.load_error
    
    def get_graph(self) -> Any:
        """Get the currently loaded graph object"""
        return self.current_graph
    
    def get_version(self) -> Optional[str]:
        """Get the currently loaded version"""
        return self.current_version
    
    def get_error(self) -> Optional[str]:
        """Get the last loading error if any"""
        return self.load_error

# Create singleton instance
rag_loader = RAGLoader()