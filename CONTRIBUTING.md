## Help us to help you!

Thank you for taking the time to contribute!

* [I want to help!](#i-want-to-help)
* [Suggesting a feature](#suggesting-a-feature)
* [Filing a bug report](#filing-a-bug-report)
* [Submitting a pull request](#submitting-a-pull-request)

## I want to help!

If you want to get involved in the development of FlowKit, the best way 
to start is to review the current documentation. Documentation includes
the examples and tutorials, as well as the Python docstrings to make 
sure they correspond to the current function/method signatures. If you 
find a mistake or want to make an addition, 
[submit a pull request](#submitting-a-pull-request) with the 
correction/addition. 

Beyond documentation, the most accessible path to start contributing to 
FlowKit is to peruse the many TODO comments in the codebase. We use the 
[PEP 350](https://www.python.org/dev/peps/pep-0350/) code tags to mark 
items that need addressing. Often times, the issues marked TODO are
relatively straight-forward compared to the 
[GitHub issues](https://github.com/whitews/FlowKit/issues) assigned to 
upcoming milestones. For example, a TODO item might mark code that should
be moved to a more appropriate class or module; or, a function/method 
that needs updating to support an additional keyword option. 

If you are considering contributing code, please submit an issue letting 
us know. Due to the complexity of flow cytometry analysis, we can likely
provide some valuable context prior to tackling an issue.

## Suggesting a feature

We cannot think of everything. If you have a good idea for a feature, 
then please let us know!

Feature suggestions are embraced, but are not guaranteed to be 
incorporated. If your suggestion is accepted, it might not be 
implemented in the next release. This is especially true for features 
that break the current version's API. Your patience is appreciated, we
do not want to break other users' code. 

When suggesting a feature, make sure to:

* Check existing issues, open and closed, to make sure it has not 
already been suggested.
* Consider if it is necessary in the library, or is an example of 
how the FlowKit library can be used. FlowKit is meant as a toolkit for
flow analysis, and not a comprehensive collection of analysis 
techniques. For these cases, consider adding the request as part of 
our collection of examples and tutorials.

## Filing a bug report

If you are seeing errors or unexpected behavior with FlowKit, then 
please submit an issue to let us know. Be as detailed as possible, and 
be ready to answer questions when we get back to you. Make sure to 
include the following in your bug report:

* Operating system & version/distribution (e.g., Windows 10, Ubuntu 18.04)
* Python version (e.g., 3.7)
* FlowKit version
* The complete traceback of the error you are seeing
* and any solutions you have tried
* And a paste/picture of the complete output from the failing 
script/library might help, too!

## Submitting a pull request

If you decide to fix a bug, even something as small as a 
single-letter typo, then great! Anything that improves the 
code/documentation for all future users is warmly welcomed.

If you decide to work on a requested feature, it is best to let us 
(and everyone else) know what you are working on to avoid any 
duplication of effort. You can do this by replying to the original 
Issue for the request.

If you want to contribute an example; go for it! We might not always 
be able to accept your code, but there is a lot to be learned from 
trying anyway and if you are new to GitHub, we are willing to guide you 
on that journey.

When contributing a new example or making a change to the code, please 
keep your code style consistent with ours. We try to stick to the PEP-8 
guidelines for Python (https://www.python.org/dev/peps/pep-0008/).

Finally, make sure all the tests pass before submitting your pull request!
No submitted code will be reviewed if there are failing tests. The 
complete test suite is run using the `run_tests.py` script found at the
root level of the repository. For code changes that result in a failing 
test that you cannot solve, submit or reply to an issue with a 
description of the problem. If you create a new function or method, 
please write new tests with coverage for all new code. 

#### Do

* Do use PEP-8 style guidelines
* Do verify that all tests pass
* Do write new tests for new functions/methods
* Do comment your code where necessary or appropriate
* Do submit only a single example/feature per pull-request
* Do include a description of what your example is expected to do

#### Do not

* Do not include any license information in your examples - our 
repositories are BSD licensed
* Do not try to do too much at once- submit one or two examples at a 
time, and be receptive to feedback
* Do not submit multiple variations of the same example, demonstrate 
one thing concisely

### If you are submitting an example

Try to do one thing, and do it concisely. Keep it simple. Do not mix 
too many ideas.

The ideal example should:

* demonstrate one idea, technique or API as concisely as possible in a 
single Python script
* *just work* when you run it. Although, sometimes configuration is 
necessary
* be well commented and attempt to teach the user how and why it works
* document any required configuration, dependencies, etc

### Licensing

When you submit code to our libraries, you implicitly and irrevocably 
agree to adopt the associated licenses, found in the file named `LICENSE`.

We use the BSD license; which permits Commercial Use, 
Modification, Distribution and Private use of our code, and therefore 
also your contributions. It also provides good compatibility with 
other licenses, and is intended to make re-use of our code as painless 
as possible for all parties.

You can learn more about BSD licenses at Wikipedia: 

https://en.wikipedia.org/wiki/BSD_licenses

### Submitting your code

Once you are ready to share your contribution with us you should submit 
it as a Pull Request.

* Be ready to receive and embrace constructive feedback.
* Be prepared for rejection; we cannot always accept contributions. If 
you are unsure, ask first!
