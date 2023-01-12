#   Copyright 2023 Joseph T. Iosue
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""warn.py.

This file contains warning functionality to standardize rcal's warnings.

"""

import warnings


class RcalWarning(UserWarning):
    """RcalWarning.

    Warning type to standardize rcal's warnings. Warn with
    ``RcalWarning.warn("message")``.

    """

    @classmethod
    def warn(cls, message):
        """warn.

        Parameters
        ----------
        message : str.
            Message to warn with.

        """
        warnings.warn(message, cls, 3)
